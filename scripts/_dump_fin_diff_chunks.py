"""提取 fin 领域 10 道差异题的检索召回片段，输出到 JSONL 供人工阅读作答。

差异题 qid 列表：fin_a_004, 005, 007, 008, 009, 011, 014, 016, 018, 020
"""
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
OUTPUT_FILE = PROJECT_ROOT / ".tmp_fin_diff_chunks.jsonl"

DIFF_QIDS = {
    "fin_a_004", "fin_a_005", "fin_a_007", "fin_a_008", "fin_a_009",
    "fin_a_011", "fin_a_014", "fin_a_016", "fin_a_018", "fin_a_019", "fin_a_020",
}

OPTION_KEYS = ("A", "B", "C", "D", "E", "F")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    # 加载预处理数据
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / "chunks_merged.jsonl")]
    print(f"Loaded {len(chunks)} chunks, {len(available_doc_ids)} docs")

    # 加载 fin 题目
    questions_file = QUESTIONS_DIR / "financial_reports_questions.json"
    all_questions = json.loads(questions_file.read_text(encoding="utf-8"))

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
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
            candidate_chunks = retrieval["candidate_chunks"]

            record = {
                "qid": qid,
                "question": raw["question"],
                "options": {k: options_dict[k] for k in OPTION_KEYS if k in options_dict},
                "answer_format": answer_format,
                "type": raw.get("type", ""),
                "doc_ids": doc_ids,
                "candidate_doc_ids": retrieval["doc_ids"],
                "chunk_count": len(candidate_chunks),
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "page_no": c.page_no,
                        "section_path": c.section_path,
                        "chunk_type": c.chunk_type,
                        "chunk_text": c.chunk_text,
                    }
                    for c in candidate_chunks
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {qid}: {len(candidate_chunks)} chunks dumped")

    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
