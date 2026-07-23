"""Answer formatting and evidence assembly.

Key improvements over original:
- Never outputs N/A — always returns a valid option letter
- Multi-choice: conservative (include insufficients if no supports)
- mcq/tf: highest-priority verdict wins
- Self-check trigger detection for low-confidence cases
"""
from __future__ import annotations

from collections import Counter

from api.models import EvidenceItem, ReasoningItem, StructuredChunk

OPTION_NAMES = ("A", "B", "C", "D", "E", "F")


def choose_answer(answer_format: str, results: list[ReasoningItem]) -> str:
    """Choose the best answer from reasoning results.

    Priority: support > insufficient > refute > default(A)

    For multi-choice: return all supports (sorted); if no supports,
    return all insufficients (conservative); if all refutes, return "A".

    For mcq/tf: return first support; if no supports, return first
    insufficient; if all refutes, return "A".

    Never returns N/A.
    """
    supports = [r.option for r in results if r.verdict == "support"]
    insufficients = [r.option for r in results if r.verdict == "insufficient"]

    if answer_format == "multi":
        if supports:
            return "".join(sorted(supports))
        if insufficients:
            # Conservative: include all uncertain options
            return "".join(sorted(insufficients))
        # All refuted — return least-bad option
        return "A"

    # mcq / tf
    if supports:
        return supports[0]
    if insufficients:
        return insufficients[0]
    return "A"


def needs_self_check(
    answer_format: str,
    results: list[ReasoningItem],
    answer: str,
) -> bool:
    """Determine if a self-check (second Qwen call) is warranted.

    Triggers:
    - All multi-choice questions (prone to over/under-selection)
    - mcq/tf with no supports (low confidence)
    """
    if not results:
        return False

    if answer_format == "multi":
        return True

    supports = [r for r in results if r.verdict == "support"]
    if not supports:
        return True

    return False


def build_evidence_items(
    results: list[ReasoningItem],
    evidence_map: dict[str, list[StructuredChunk]],
) -> list[EvidenceItem]:
    """Build deduplicated evidence items from reasoning results."""
    evidence_items: list[EvidenceItem] = []
    for result in results:
        for chunk in evidence_map.get(result.option, []):
            evidence_items.append(
                EvidenceItem(
                    docId=chunk.doc_id,
                    pageNo=chunk.page_no,
                    clauseNo=chunk.clause_no,
                    quotedText=chunk.chunk_text,
                    supportsOption=[result.option],
                    reasoning=result.reasoning,
                )
            )
    unique_items: list[EvidenceItem] = []
    seen: Counter[str] = Counter()
    for item in evidence_items:
        key = f"{item.doc_id}:{item.page_no}:{item.quoted_text[:50]}"
        if seen[key]:
            continue
        seen[key] += 1
        unique_items.append(item)
    return unique_items[:8]
