from __future__ import annotations

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.planner import activate_rewrite, build_plan, next_rewrite_from_feedback
from api.services.reasoner import reason_options
from api.services.retrieval_evaluator import evaluate_retrieval_quality
from api.services.retriever import rank_chunks, rank_documents


def _resolve_doc_ids(
    payload: RunQuestionTaskRequest,
    *,
    available_doc_ids: set[str],
    structured_chunks: list[StructuredChunk],
    plan: dict[str, object],
    previous_doc_ids: list[str],
) -> list[str]:
    if payload.mode == "A":
        return [doc_id for doc_id in (payload.doc_ids or []) if doc_id in available_doc_ids]

    ranked_docs = [doc_id for doc_id in rank_documents(payload.question, structured_chunks, plan) if doc_id in available_doc_ids]
    if plan.get("requires_multi_doc"):
        combined = []
        for doc_id in [*previous_doc_ids, *ranked_docs]:
            if doc_id not in combined:
                combined.append(doc_id)
        return combined[:8]
    return ranked_docs


def run_retrieval_loop(
    payload: RunQuestionTaskRequest,
    *,
    available_doc_ids: set[str],
    structured_chunks: list[StructuredChunk],
) -> dict[str, object]:
    base_plan = build_plan(payload)
    current_plan = activate_rewrite(base_plan, "base")
    previous_doc_ids: list[str] = []
    loop_logs: list[str] = []
    best_round: dict[str, object] | None = None
    max_iterations = int(base_plan.get("max_iterations", 3) or 3)

    for iteration in range(1, max_iterations + 1):
        current_plan["iteration"] = iteration
        current_doc_ids = _resolve_doc_ids(
            payload,
            available_doc_ids=available_doc_ids,
            structured_chunks=structured_chunks,
            plan=current_plan,
            previous_doc_ids=previous_doc_ids,
        )
        previous_doc_ids = current_doc_ids
        candidate_chunks = rank_chunks(
            payload.question,
            payload.options,
            structured_chunks,
            current_doc_ids or None,
            current_plan,
        )
        reasoning_results, evidence_map = reason_options(
            payload.options, candidate_chunks, payload.question, payload.answer_format,
        )
        evaluation = evaluate_retrieval_quality(
            question=payload.question,
            options=payload.options,
            plan=current_plan,
            candidate_chunks=candidate_chunks,
            reasoning_results=reasoning_results,
            candidate_doc_ids=current_doc_ids,
        )
        round_result = {
            "plan": dict(current_plan),
            "doc_ids": list(current_doc_ids),
            "candidate_chunks": candidate_chunks,
            "reasoning_results": reasoning_results,
            "evidence_map": evidence_map,
            "evaluation": evaluation,
        }
        if best_round is None or float(evaluation["quality_score"]) >= float(best_round["evaluation"]["quality_score"]):
            best_round = round_result

        loop_logs.extend(
            [
                f"react_round={iteration}",
                f"react_rewrite={current_plan.get('active_rewrite_name', 'base')}",
                f"react_doc_count={len(current_doc_ids)}",
                f"react_chunk_count={len(candidate_chunks)}",
                f"react_quality={evaluation['quality_score']}",
                f"react_quality_label={evaluation['quality_label']}",
                f"react_decision={evaluation['decision']}",
                f"react_next_action={evaluation['next_action']}",
                f"react_failures={','.join(evaluation['failure_reasons']) if evaluation['failure_reasons'] else 'none'}",
            ]
        )

        if evaluation["decision"] == "accept" or iteration >= max_iterations:
            chosen = round_result if evaluation["decision"] == "accept" else best_round or round_result
            return {
                **chosen,
                "base_plan": base_plan,
                "loop_logs": loop_logs,
            }

        current_plan = next_rewrite_from_feedback(base_plan, evaluation, payload.options)

    chosen = best_round or {
        "plan": current_plan,
        "doc_ids": [],
        "candidate_chunks": [],
        "reasoning_results": [],
        "evidence_map": {},
        "evaluation": {"quality_score": 0.0, "quality_label": "low", "decision": "retry", "next_action": "accept"},
    }
    return {
        **chosen,
        "base_plan": base_plan,
        "loop_logs": loop_logs,
    }
