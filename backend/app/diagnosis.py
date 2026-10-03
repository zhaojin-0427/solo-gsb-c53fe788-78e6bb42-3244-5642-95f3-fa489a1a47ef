"""Draft reachability diagnosis (草稿可达性诊断).

A read-only, deterministic analyzer the presenter runs on a *draft* script
version before publishing it. The frozen definition stored on the version row
is the only input; the report answers four questions:

* **可达阶段的最短选择路径** — which stages are reachable from the initial
  stage, and the shortest choice path (fewest settlements) leading to each;
* **不可达阶段** — stages no combination of submissions/timeouts can reach;
* **被遮蔽的条件转移** — conditional transitions that can never fire because
  an earlier rule always wins first (or whose condition can never hold);
* **无法通向终局的阶段** — stages from which no outcome ends the drill.

Adjudication order is identical to settlement, because the analysis literally
calls the engine's ``evaluate_trace`` on representative choice combinations:

1. an explicit ``next`` on a submitted option wins; when several roles submit
   options with explicit jumps, the first role in the stage's role
   configuration order decides;
2. otherwise the first conditional transition (YAML order) whose ``when`` is
   fully satisfied wins;
3. otherwise ``default_next`` (``None`` ends the drill).

Roles that never submit are part of the model: every deciding role may also
time out (choice ``None``), so the all-timeout outcome is explored as well.
Equal-length paths are chosen by definition order (edges are enumerated in
adjudication order and BFS discovers stages in that order), and a cycle that
has an exit is never reported as a dead end — only stages from which *no*
outcome reaches the terminal are flagged.

Nothing here writes anything; re-running the diagnosis on the same draft
version reproduces the same report.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from .definitions import DrillDef, Stage, Transition
from .engine import evaluate_trace

END_LABEL = "演练结束"


# --------------------------------------------------------------------------- #
# per-stage outcome edges
# --------------------------------------------------------------------------- #
def _full_choices(stage: Stage, assign: dict[str, Any]) -> dict[str, Any]:
    """Expand a partial assignment to the choices mapping the engine sees at
    settlement: one entry per deciding role, ``None`` for non-submitters."""
    return {r: assign.get(r) for r in stage.options}


def _match_problem(stage: Stage, t: Transition) -> str | None:
    """Why a transition's ``when`` can never be satisfied, or None.

    A ``None`` requirement matches "the role did not submit" (exactly like
    the settlement evaluator); any other value must equal an existing option
    id of a role that actually decides in this stage.
    """
    for role, want in t.when.items():
        if want is None:
            continue
        if role not in stage.options:
            return f"角色「{role}」在本阶段没有决策项"
        if not any(o.id == want for o in stage.options[role]):
            return f"角色「{role}」没有选项「{want}」"
    return None


def _default_witness(stage: Stage) -> dict[str, Any] | None:
    """A choice assignment that falls through every rule to ``default_next``,
    or None when the default outcome can never happen.

    Only submissions of options *without* an explicit jump (or no submission
    at all) can reach the default, so the search space per role is
    ``{None} ∪ no-jump options``. A depth-first cover check prunes whole
    subtrees as soon as a transition matches, preferring unsubmitted roles
    first, which yields the most timeout-flavoured witness.
    """
    roles = list(stage.options.keys())
    values = {
        r: [None] + [o.id for o in olist if o.next is None]
        for r, olist in stage.options.items()
    }
    transitions = stage.transitions
    assign: dict[str, Any] = {}

    def falsified(t: Transition) -> bool:
        for role, want in t.when.items():
            if role in assign:
                if assign[role] != want:
                    return True
            elif role not in values:
                # A non-deciding role is always None at settlement time.
                if want is not None:
                    return True
        return False

    def matched(t: Transition) -> bool:
        if falsified(t):
            return False
        return all((r in assign) or (r not in values) for r in t.when)

    def covered(i: int) -> bool:
        """True iff every completion of the partial assignment hits a rule."""
        if any(matched(t) for t in transitions):
            return True
        if i == len(roles):
            return False
        r = roles[i]
        for v in values[r]:
            assign[r] = v
            if not covered(i + 1):
                return False  # assign keeps the witness prefix
            del assign[r]
        return True

    if covered(0):
        return None
    return dict(assign)


def _stage_edges(
    d: DrillDef, stage: Stage
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All outcomes a stage can produce, enumerated in adjudication order,
    plus the shadowed-transition findings.

    Every edge carries a minimal representative assignment that is verified
    against the engine's own ``evaluate_trace``, so the diagnosis and the
    settlement can never disagree about who wins.
    """
    edges: list[dict[str, Any]] = []
    shadowed: list[dict[str, Any]] = []

    # 1. explicit option jumps: stage role configuration order, option order.
    for role_id, olist in stage.options.items():
        for opt in olist:
            if opt.next is None:
                continue
            assign = {role_id: opt.id}
            outcome, detail = evaluate_trace(stage, _full_choices(stage, assign))
            if detail.get("via") != "option" or outcome != opt.next:
                continue  # unreachable by construction; stay honest anyway
            edges.append({
                "via": "option",
                "target": outcome,
                "assign": assign,
                "role_id": role_id,
                "option_id": opt.id,
            })

    # 2. conditional transitions in YAML order. A transition can fire iff it
    #    wins on its *minimal* satisfying assignment: supersets only add
    #    higher-priority competition (more submitted options with explicit
    #    jumps, more earlier transitions fully matched), never remove it.
    for idx, t in enumerate(stage.transitions):
        problem = _match_problem(stage, t)
        if problem is not None:
            shadowed.append({
                "stage_id": stage.id,
                "stage_title": stage.title,
                "index": idx,
                "when": dict(t.when),
                "next": t.next,
                "reason": "unmatchable",
                "reason_label": f"条件永远无法成立：{problem}",
                "shadowed_by": None,
            })
            continue
        assign = {r: w for r, w in t.when.items()}
        outcome, detail = evaluate_trace(stage, _full_choices(stage, assign))
        if detail.get("via") == "transition" and detail.get("index") == idx:
            edges.append({
                "via": "transition",
                "target": outcome,
                "assign": assign,
                "transition_index": idx,
            })
            continue
        if detail.get("via") == "option":
            rid, oid = detail["role_id"], detail["option_id"]
            role = d.roles.get(rid)
            opt = next(
                (o for o in stage.options.get(rid, []) if o.id == oid), None
            )
            reason = "shadowed_by_option"
            label = (
                f"被角色「{role.name if role else rid}」选项"
                f"「{opt.text if opt else oid}」的显式跳转遮蔽："
                "条件一旦成立，该显式跳转必先生效"
            )
            by: dict[str, Any] | None = {
                "via": "option",
                "role_id": rid,
                "role_name": role.name if role else rid,
                "option_id": oid,
                "option_text": opt.text if opt else oid,
            }
        else:
            j = detail.get("index")
            reason = "shadowed_by_transition"
            label = (
                f"被排在前面的条件转移 #{j} 遮蔽："
                f"本规则条件成立时，规则 #{j} 必先命中"
            )
            by = {"via": "transition", "index": j}
        shadowed.append({
            "stage_id": stage.id,
            "stage_title": stage.title,
            "index": idx,
            "when": dict(t.when),
            "next": t.next,
            "reason": reason,
            "reason_label": label,
            "shadowed_by": by,
        })

    # 3. default jump: only when some assignment falls through every rule.
    witness = _default_witness(stage)
    if witness is not None:
        outcome, detail = evaluate_trace(stage, _full_choices(stage, witness))
        if detail.get("via") == "default":
            edges.append({"via": "default", "target": outcome, "assign": witness})
    return edges, shadowed


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def _role_name(d: DrillDef, role_id: str) -> str:
    r = d.roles.get(role_id)
    return r.name if r else role_id


def _option_text(stage: Stage, role_id: str, option_id: str) -> str:
    for o in stage.options.get(role_id, []):
        if o.id == option_id:
            return o.text
    return option_id


def _step_json(d: DrillDef, stage: Stage, edge: dict[str, Any]) -> dict[str, Any]:
    """One hop of a shortest path: at ``stage``, these submissions (or
    timeouts) lead via this rule to ``edge['target']``."""
    assign = edge["assign"]
    choices = [
        {
            "role_id": r,
            "role_name": _role_name(d, r),
            "option_id": o,
            "option_text": _option_text(stage, r, str(o)),
        }
        for r, o in assign.items()
        if o is not None and r in stage.options
    ]
    unsubmitted = [
        {"role_id": r, "role_name": _role_name(d, r)}
        for r in stage.options
        if assign.get(r) is None
    ]
    via = edge["via"]
    if via == "option":
        rid, oid = edge["role_id"], edge["option_id"]
        via_label = (
            f"角色「{_role_name(d, rid)}」选项"
            f"「{_option_text(stage, rid, oid)}」的显式跳转"
        )
    elif via == "transition":
        via_label = f"条件转移 #{edge['transition_index']}"
    else:
        via_label = "默认跳转（无任何规则命中）"
    target = edge["target"]
    target_stage = d.stages.get(target) if target is not None else None
    return {
        "stage_id": stage.id,
        "stage_title": stage.title,
        "via": via,
        "via_label": via_label,
        "transition_index": edge.get("transition_index"),
        "choices": choices,
        "unsubmitted": unsubmitted,
        "next": target,
        "next_title": target_stage.title if target_stage else None,
        "next_label": target_stage.title if target_stage else END_LABEL,
    }


def diagnose_definition(d: DrillDef) -> dict[str, Any]:
    """Build the reachability report for one frozen draft definition."""
    stage_list = list(d.stages.values())  # definition (YAML) order
    edges: dict[str, list[dict[str, Any]]] = {}
    shadowed: list[dict[str, Any]] = []
    for st in stage_list:
        st_edges, st_shadowed = _stage_edges(d, st)
        edges[st.id] = st_edges
        shadowed.extend(st_shadowed)

    # Shortest paths: BFS from the initial stage. Edges are enumerated in
    # adjudication order and the queue is FIFO, so the first discovery of a
    # stage is both shortest (fewest settlements) and, among equal-length
    # paths, the one following definition order.
    dist = {d.initial_stage: 0}
    prev: dict[str, tuple[str, dict[str, Any]]] = {}
    queue: deque[str] = deque([d.initial_stage])
    while queue:
        sid = queue.popleft()
        for e in edges[sid]:
            tgt = e["target"]
            if tgt is None or tgt in dist:
                continue
            dist[tgt] = dist[sid] + 1
            prev[tgt] = (sid, e)
            queue.append(tgt)

    # Stages that can reach the terminal: reverse reachability from the
    # virtual end node. A cycle with an exit reaches it and is never
    # misreported; only stages whose every outcome stays inside a dead
    # region are flagged.
    can_finish: set[str] = set()
    reverse: dict[str, list[str]] = {}
    for st in stage_list:
        for e in edges[st.id]:
            if e["target"] is None:
                can_finish.add(st.id)
            else:
                reverse.setdefault(e["target"], []).append(st.id)
    rqueue: deque[str] = deque(can_finish)
    while rqueue:
        sid = rqueue.popleft()
        for src in reverse.get(sid, []):
            if src not in can_finish:
                can_finish.add(src)
                rqueue.append(src)

    def path_to(sid: str) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        cur = sid
        while cur != d.initial_stage:
            parent, edge = prev[cur]
            steps.append(_step_json(d, d.stages[parent], edge))
            cur = parent
        steps.reverse()
        return steps

    stages_json: list[dict[str, Any]] = []
    for st in stage_list:
        reachable = st.id in dist
        stages_json.append({
            "id": st.id,
            "title": st.title,
            "initial": st.id == d.initial_stage,
            "reachable": reachable,
            "distance": dist.get(st.id),
            "can_finish": st.id in can_finish,
            "path": path_to(st.id) if reachable else None,
        })

    unreachable = [
        {"id": st.id, "title": st.title, "can_finish": st.id in can_finish}
        for st in stage_list
        if st.id not in dist
    ]
    no_exit = [
        {"id": st.id, "title": st.title, "reachable": st.id in dist}
        for st in stage_list
        if st.id not in can_finish
    ]
    summary = {
        "stages": len(stage_list),
        "reachable": len(dist),
        "unreachable": len(unreachable),
        "shadowed_transitions": len(shadowed),
        "no_exit": len(no_exit),
        "all_reachable": not unreachable,
        "all_can_finish": not no_exit,
    }
    return {
        "kind": "draft_reachability",
        "initial_stage": d.initial_stage,
        "summary": summary,
        "stages": stages_json,
        "unreachable": unreachable,
        "no_exit": no_exit,
        "shadowed_transitions": shadowed,
    }
