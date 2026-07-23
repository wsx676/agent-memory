"""提取 fin 领域9道1200cap错题的 reasoning_hints，分析锚定效应。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.retrieval_loop import run_retrieval_loop

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"

DIFF_QIDS = {
    "fin_a_004", "fin_a_008", "fin_a_009", "fin_a_011",
    "fin_a_014", "fin_a_016", "fin_a_018", "fin_a_019", "fin_a_020",
}

OPTION_KEYS = ("A", "B", "C", "D", "E", "F")
OUTPUT_FILE = PROJECT_ROOT / ".tmp_fin_diff_hints.txt"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / "chunks_merged.jsonl")]
    print(f"Loaded {len(chunks)} chunks, {len(available_doc_ids)} docs")

    questions_file = QUESTIONS_DIR / "financial_reports_questions.json"
    all_questions = json.loads(questions_file.read_text(encoding="utf-8"))

    lines: list[str] = []
    for raw in all_questions:
        qid = raw["qid"]
        if qid not in DIFF_QIDS:
            continue

        options_dict = raw.get("options", {})
        options = [options_dict[k] for k in OPTION_KEYS if k in options_dict]
        answer_format = raw["answer_format"]
        doc_ids = raw.get("doc_ids", [])

        request = RunQuestionTaskRequest(
            mode="A",
            qid=qid,
            question=raw["question"],
            options=options,
            answerFormat=answer_format,
            docIds=doc_ids,
        )

        retrieval = run_retrieval_loop(request, available_doc_ids=available_doc_ids, structured_chunks=chunks)
        reasoning_results = retrieval.get("reasoning_results", [])

        lines.append(f"\n{'='*80}")
        lines.append(f"qid: {qid}  answer_format: {answer_format}")
        lines.append(f"question: {raw['question'][:100]}...")
        lines.append(f"options:")
        for k in OPTION_KEYS:
            if k in options_dict:
                lines.append(f"  {k}: {options_dict[k]}")

        lines.append(f"\nreasoning_hints:")
        if not reasoning_results:
            lines.append("  (无 - 单选题不注入hints)")
        else:
            for item in reasoning_results:
                safe_reasoning = item.reasoning[:200].replace('\n', ' ')
                lines.append(f"  选项{item.option}: verdict={item.verdict}")
                lines.append(f"    reasoning: {safe_reasoning}")

        from api.services.qwen_client import _format_prejudgment
        from api.models import ReasoningItem
        hints = [ReasoningItem(option=r.option, verdict=r.verdict, reasoning=r.reasoning) for r in reasoning_results]
        prejudgment = _format_prejudgment(hints)
        if prejudgment:
            lines.append(f"\nLLM实际看到的prejudgment文本:")
            lines.append(prejudgment)

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
