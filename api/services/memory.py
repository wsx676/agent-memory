from __future__ import annotations

from api.models import ReasoningItem, StructuredChunk


def build_memory_ledger(qid: str, question: str, evidence_map: dict[str, list[StructuredChunk]], results: list[ReasoningItem]) -> dict[str, object]:
    facts = []
    for option, chunks in evidence_map.items():
        if not chunks:
            continue
        facts.append(
            {
                "option": option,
                "docIds": sorted({chunk.doc_id for chunk in chunks}),
                "pageNos": sorted({chunk.page_no for chunk in chunks}),
            }
        )
    return {
        "qid": qid,
        "question_focus": [question[:80]],
        "facts": facts,
        "option_status": {result.option: result.verdict for result in results},
        "missing_fields": [result.option for result in results if result.verdict == "insufficient"],
        "followup_needed": any(result.verdict == "insufficient" for result in results),
    }
