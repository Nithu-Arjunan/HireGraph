import json
import time
from pathlib import Path
from typing import Any

from langgraph.types import Command

from src.graph import build_hiregraph
from src.graph_print import save_graph_png


SCENARIOS = [
    {
        "id": "strong",
        "label": "Strong Candidate",
        "resume_path": "sample_data/resume/resume_priya.md",
        "jd_path": "sample_data/jd/jd_junior_data.md",
        "reviewer_decision": "approved",
    },
    {
        "id": "borderline",
        "label": "Borderline Candidate",
        "resume_path": "sample_data/resume/resume_eitan.md",
        "jd_path": "sample_data/jd/jd_junior_data.md",
        "reviewer_decision": "approved",
    },
    {
        "id": "weak",
        "label": "Weak Candidate",
        "resume_path": "sample_data/resume/resume_mira.md",
        "jd_path": "sample_data/jd/jd_junior_data.md",
        "reviewer_decision": "rejected",
    },
]

ACTIVE_DEMO_SESSION: dict[str, Any] = {}


def _score_items(items: list[dict[str, Any]], name_key: str) -> list[dict[str, Any]]:
    score_items = []
    for item in items:
        score_items.append(
            {
                "name": item.get(name_key, "unknown"),
                "score": item.get("score", 0),
                "evidence": item.get("evidence", ""),
                "reasoning": item.get("reasoning", ""),
            }
        )
    return score_items


def _audit_rows(items: list[dict[str, Any]], elapsed_ms: int) -> list[dict[str, Any]]:
    if not items:
        return [
            {
                "node": "graph",
                "verdict": "completed",
                "message": "Graph completed without node-level audit rows.",
                "timing_ms": elapsed_ms,
            }
        ]

    per_node_ms = max(1, elapsed_ms // len(items))
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "node": item.get("node", "unknown"),
                "verdict": item.get("status", item.get("verdict", "completed")),
                "message": item.get("message", ""),
                "timing_ms": item.get("timing_ms", per_node_ms),
                "order": index + 1,
            }
        )
    return rows


def build_decision_object(
    scenario_id: str,
    state: dict[str, Any],
    elapsed_ms: int,
    label: str | None = None,
    run_status: str = "completed",
) -> dict[str, Any]:
    email_draft = state.get("draft_email") or state.get("rejection_email") or ""
    action_status = {
        "email_sent": bool(state.get("email_sent")),
        "ats_updated": bool(state.get("ats_updated")),
        "rejection_logged": bool(state.get("rejection_logged")),
        "compensation_done": bool(state.get("compensation_done")),
    }

    return {
        "scenario": scenario_id,
        "label": label or scenario_id.title(),
        "candidate_name": state.get("candidate_name", "Candidate"),
        "candidate_email": state.get("candidate_email"),
        "elapsed_ms": elapsed_ms,
        "run_status": run_status,
        "interrupt": _serialize_interrupt(state.get("__interrupt__")),
        "decision": {
            "scorecard": {
                "final_score": state.get("final_score", 0),
                "summary": state.get("score_summary", {}),
                "skill_scores": _score_items(state.get("skill_evaluations", []), "skill"),
                "dimension_scores": _score_items(
                    state.get("dimension_evaluations", []),
                    "dimension",
                ),
            },
            "recommendation": state.get("recommendation", "unknown"),
            "recommendation_reasoning": state.get("recommendation_reasoning", ""),
            "email_draft": email_draft,
            "action_status": action_status,
            "audit_trail": _audit_rows(state.get("audit_trail", []), elapsed_ms),
        },
    }


def _serialize_interrupt(interrupt_value: Any) -> dict[str, Any] | None:
    if not interrupt_value:
        return None

    first_interrupt = interrupt_value[0] if isinstance(interrupt_value, list) else interrupt_value
    value = getattr(first_interrupt, "value", None)
    if value is None and isinstance(first_interrupt, dict):
        value = first_interrupt.get("value", first_interrupt)
    if isinstance(value, dict):
        return {"value": value}
    return {"value": {"message": str(value or first_interrupt)}}


def build_scoreboard_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"advance": 0, "reject": 0, "borderline": 0}
    scores = []

    for decision in decisions:
        recommendation = decision.get("decision", {}).get("recommendation")
        if recommendation in counts:
            counts[recommendation] += 1
        score = decision.get("decision", {}).get("scorecard", {}).get("final_score")
        if isinstance(score, int | float):
            scores.append(score)

    average_score = round(sum(scores) / len(scores), 2) if scores else 0
    return {
        "total": len(decisions),
        "advance": counts["advance"],
        "reject": counts["reject"],
        "borderline": counts["borderline"],
        "average_score": average_score,
    }


def run_hiregraph_scenarios() -> dict[str, Any]:
    graph = build_hiregraph()
    save_graph_png(graph, "hiregraph")

    decisions = []
    pending_reviews = {}
    for scenario in SCENARIOS:
        start = time.perf_counter()
        config = {"configurable": {"thread_id": f"demo-{scenario['id']}"}}
        result = graph.invoke(
            {
                "resume_path": scenario["resume_path"],
                "jd_path": scenario["jd_path"],
            },
            config=config,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        run_status = "interrupted" if "__interrupt__" in result else "completed"
        if run_status == "interrupted":
            pending_reviews[scenario["id"]] = {
                "config": config,
                "scenario": scenario,
                "started_at": start,
            }
        decisions.append(
            build_decision_object(
                scenario["id"],
                result,
                elapsed_ms=elapsed_ms,
                label=scenario["label"],
                run_status=run_status,
            )
        )

    summary = build_scoreboard_summary(decisions)
    payload = {
        "graph_png": "graph_out/hiregraph.png",
        "graph_mmd": "graph_out/hiregraph.mmd",
        "scoreboard": summary,
        "decisions": decisions,
        "pending_reviews": list(pending_reviews.keys()),
    }
    ACTIVE_DEMO_SESSION.clear()
    ACTIVE_DEMO_SESSION.update(
        {
            "graph": graph,
            "payload": payload,
            "pending_reviews": pending_reviews,
        }
    )
    write_scoreboard_summary(payload)
    return payload


def resume_hiregraph_scenario(
    scenario_id: str,
    reviewer_decision: str,
    notes: str = "",
) -> dict[str, Any]:
    pending_reviews = ACTIVE_DEMO_SESSION.get("pending_reviews", {})
    if scenario_id not in pending_reviews:
        raise ValueError(f"No pending human review for scenario: {scenario_id}")

    graph = ACTIVE_DEMO_SESSION["graph"]
    review = pending_reviews[scenario_id]
    reviewer_input = {
        "decision": reviewer_decision,
        "notes": notes,
    }
    result = graph.invoke(Command(resume=reviewer_input), config=review["config"])
    elapsed_ms = int((time.perf_counter() - review["started_at"]) * 1000)
    updated_decision = build_decision_object(
        scenario_id,
        result,
        elapsed_ms=elapsed_ms,
        label=review["scenario"]["label"],
        run_status="completed",
    )

    payload = ACTIVE_DEMO_SESSION["payload"]
    payload["decisions"] = [
        updated_decision if item["scenario"] == scenario_id else item
        for item in payload["decisions"]
    ]
    pending_reviews.pop(scenario_id)
    payload["pending_reviews"] = list(pending_reviews.keys())
    payload["scoreboard"] = build_scoreboard_summary(payload["decisions"])
    write_scoreboard_summary(payload)
    return payload


def write_scoreboard_summary(payload: dict[str, Any]) -> None:
    out_dir = Path("graph_out")
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / "scoreboard_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n========== SCOREBOARD SUMMARY ==========")
    print(json.dumps(payload["scoreboard"], indent=2))
    for decision in payload["decisions"]:
        scorecard = decision["decision"]["scorecard"]
        print(
            f"{decision['label']}: "
            f"{decision['decision']['recommendation']} "
            f"({scorecard['final_score']})"
        )
