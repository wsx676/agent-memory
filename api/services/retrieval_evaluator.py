from __future__ import annotations

from collections import Counter

from api.models import ReasoningItem, StructuredChunk
from api.services.retriever import score_overlap


def _joined_chunk_text(chunks: list[StructuredChunk]) -> str:
    return " ".join(
        " ".join((chunk.title or "", chunk.section_path or "", chunk.clause_no or "", chunk.chunk_text or ""))
        for chunk in chunks
    )


def evaluate_retrieval_quality(
    *,
    question: str,
    options: list[str],
    plan: dict[str, object],
    candidate_chunks: list[StructuredChunk],
    reasoning_results: list[ReasoningItem],
    candidate_doc_ids: list[str],
) -> dict[str, object]:
    if not candidate_chunks:
        return {
            "quality_score": 0.0,
            "quality_label": "low",
            "decision": "retry",
            "next_action": "broaden_docs",
            "failure_reasons": ["candidate_chunks_empty"],
            "insufficient_options": [result.option for result in reasoning_results],
        }

    joined_text = _joined_chunk_text(candidate_chunks)
    doc_coverage = len({chunk.doc_id for chunk in candidate_chunks})
    section_coverage = len({chunk.section_path or chunk.title or chunk.chunk_id for chunk in candidate_chunks})
    question_overlap = max(score_overlap(question, chunk.chunk_text) for chunk in candidate_chunks)

    numbers = [str(item) for item in plan.get("numbers", []) if str(item)]
    clause_refs = [str(item) for item in plan.get("clause_refs", []) if str(item)]
    focus_terms = [str(item) for item in plan.get("focus_terms", []) if str(item)]
    number_hit_rate = (
        sum(1 for item in numbers if item in joined_text) / len(numbers)
        if numbers
        else 1.0
    )
    clause_hit_rate = (
        sum(1 for item in clause_refs if item in joined_text) / len(clause_refs)
        if clause_refs
        else 1.0
    )
    focus_term_hit_rate = (
        sum(1 for item in focus_terms if item in joined_text) / len(focus_terms)
        if focus_terms
        else 1.0
    )

    verdict_counts = Counter(result.verdict for result in reasoning_results)
    insufficient_options = [result.option for result in reasoning_results if result.verdict == "insufficient"]
    supported_or_refuted = verdict_counts.get("support", 0) + verdict_counts.get("refute", 0)
    expected_doc_coverage = int(((plan.get("quality_gate") or {}).get("min_doc_coverage", 1)) or 1)

    quality_score = (
        min(len(candidate_chunks), 6) / 6 * 0.18
        + min(doc_coverage, expected_doc_coverage) / max(expected_doc_coverage, 1) * 0.20
        + min(section_coverage, 4) / 4 * 0.14
        + min(question_overlap, 1.0) * 0.12
        + min(number_hit_rate, 1.0) * 0.12
        + min(clause_hit_rate, 1.0) * 0.12
        + min(focus_term_hit_rate, 1.0) * 0.06
        + min(supported_or_refuted / max(len(options), 1), 1.0) * 0.06
    )

    failure_reasons: list[str] = []
    next_action = "accept"
    if doc_coverage < expected_doc_coverage:
        failure_reasons.append("doc_coverage_low")
        next_action = "comparison_focus"
    if clause_refs and clause_hit_rate < 1.0:
        failure_reasons.append("clause_coverage_low")
        next_action = "clause_focus"
    if numbers and number_hit_rate < 1.0:
        failure_reasons.append("number_coverage_low")
        next_action = "numbers_focus"
    # insufficient 触发条件按题型区分：
    # - 多选题：只要有 1 个 insufficient 就 retry（错一个全错，不能放过）
    # - 单选/判断题：>=2 个 insufficient 才 retry（避免过度重试简单题）
    insuf_count = verdict_counts.get("insufficient", 0)
    answer_format = str(plan.get("answer_format", "single"))
    insuf_threshold = 1 if answer_format == "multi" else max(1, len(options) // 2)
    if insuf_count >= insuf_threshold:
        failure_reasons.append("insufficient_options_high")
        next_action = "option_split"
    if section_coverage <= 1 and len(candidate_chunks) >= 2 and plan.get("question_type") in {"comparison", "calculation"}:
        failure_reasons.append("section_diversity_low")
        next_action = "diversify_sections"
    if not failure_reasons and focus_term_hit_rate < 0.5:
        failure_reasons.append("focus_term_coverage_low")
        next_action = "section_focus"

    quality_label = "high" if quality_score >= 0.72 else "medium" if quality_score >= 0.5 else "low"
    # 有 failure_reasons 就 retry，不再因为 quality_label=high 而跳过
    # （之前 high 会强制 accept，导致 insufficient 题也被放过）
    decision = "accept" if not failure_reasons else "retry"
    if not candidate_doc_ids:
        failure_reasons.append("candidate_doc_ids_empty")
        decision = "retry"
        next_action = "broaden_docs"

    return {
        "quality_score": round(quality_score, 4),
        "quality_label": quality_label,
        "decision": decision,
        "next_action": next_action,
        "failure_reasons": failure_reasons,
        "insufficient_options": insufficient_options,
        "doc_coverage": doc_coverage,
        "section_coverage": section_coverage,
        "number_hit_rate": round(number_hit_rate, 4),
        "clause_hit_rate": round(clause_hit_rate, 4),
        "focus_term_hit_rate": round(focus_term_hit_rate, 4),
    }
