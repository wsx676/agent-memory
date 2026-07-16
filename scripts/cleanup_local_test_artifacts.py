from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DOCS = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed" / "documents.jsonl"

DOCUMENTS_FILE = DATA_DIR / "documents.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
TASKS_FILE = DATA_DIR / "tasks.json"
RESULTS_FILE = DATA_DIR / "results.json"
EVIDENCE_FILE = DATA_DIR / "evidence.json"
ANSWER_FILE = DATA_DIR / "answer.csv"

OFFICIAL_QID_PATTERN = re.compile(r"^(fc|fin|ins|reg|res)_a_\d+$", re.IGNORECASE)


def read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_array(path: Path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def is_pytest_temp_path(source_path: str) -> bool:
    normalized = normalize_path(source_path)
    return "/appdata/local/temp/pytest-of-" in normalized or "/pytest-of-" in normalized


def is_official_qid(qid: str) -> bool:
    return bool(OFFICIAL_QID_PATTERN.fullmatch(qid))


def is_test_like_qid(qid: str) -> bool:
    lowered = qid.lower()
    return lowered.startswith("q-") or lowered.startswith("test-") or lowered.startswith("debug-") or lowered.startswith("loop-")


def doc_ids_from_result(result: dict[str, Any]) -> set[str]:
    doc_ids = {str(item.get("docId")) for item in result.get("evidence", []) if item.get("docId")}
    ledger = result.get("ledger", {})
    for fact in ledger.get("facts", []):
        doc_ids.update(str(doc_id) for doc_id in fact.get("docIds", []) if doc_id)
    return doc_ids


def doc_ids_from_evidence_row(row: dict[str, Any]) -> set[str]:
    return {str(item.get("docId")) for item in row.get("evidence_retrieval", []) if item.get("docId")}


def rewrite_answer_csv(path: Path, removed_qids: set[str]) -> dict[str, int]:
    if not path.exists():
        return {"removed_answer_rows": 0, "remaining_answer_rows": 0}

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"removed_answer_rows": 0, "remaining_answer_rows": 0}

    header = rows[0]
    data_rows = [row for row in rows[1:] if row]
    original_rows = [row for row in data_rows if row[0] != "summary"]
    kept_rows = [row for row in original_rows if row[0] not in removed_qids]

    total_prompt = sum(int(row[2]) for row in kept_rows)
    total_completion = sum(int(row[3]) for row in kept_rows)
    total_tokens = sum(int(row[4]) for row in kept_rows)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["summary", "", total_prompt, total_completion, total_tokens])
        writer.writerows(kept_rows)

    return {
        "removed_answer_rows": len(original_rows) - len(kept_rows),
        "remaining_answer_rows": len(kept_rows),
    }


def main() -> None:
    official_doc_ids = {str(item["doc_id"]) for item in read_jsonl(PREPROCESSED_DOCS) if item.get("doc_id")}

    documents = read_json_array(DOCUMENTS_FILE)
    removed_doc_ids = {
        str(item.get("documentId"))
        for item in documents
        if is_pytest_temp_path(str(item.get("sourcePath", ""))) and item.get("documentId")
    }
    kept_documents = [item for item in documents if str(item.get("documentId")) not in removed_doc_ids]

    chunks = read_json_array(CHUNKS_FILE)
    kept_chunks = [item for item in chunks if str(item.get("docId")) not in removed_doc_ids]

    results = read_json_array(RESULTS_FILE)
    removed_result_task_ids: set[str] = set()
    removed_qids: set[str] = set()
    kept_results: list[dict[str, Any]] = []
    for row in results:
        qid = str(row.get("qid", ""))
        referenced_doc_ids = doc_ids_from_result(row)
        references_removed_docs = bool(referenced_doc_ids & removed_doc_ids)
        references_official_docs = bool(referenced_doc_ids & official_doc_ids)
        should_remove = references_removed_docs or (is_test_like_qid(qid) and not references_official_docs and not is_official_qid(qid))
        if should_remove:
            if row.get("taskId"):
                removed_result_task_ids.add(str(row["taskId"]))
            if qid and not is_official_qid(qid):
                removed_qids.add(qid)
            continue
        kept_results.append(row)

    tasks = read_json_array(TASKS_FILE)
    kept_tasks = []
    for row in tasks:
        task_type = str(row.get("type", ""))
        if task_type == "preprocess" and str(row.get("documentId")) in removed_doc_ids:
            continue
        if task_type == "question" and (str(row.get("taskId")) in removed_result_task_ids or str(row.get("qid")) in removed_qids):
            continue
        kept_tasks.append(row)

    evidence_rows = read_json_array(EVIDENCE_FILE)
    kept_evidence = []
    for row in evidence_rows:
        qid = str(row.get("qid", ""))
        referenced_doc_ids = doc_ids_from_evidence_row(row)
        references_removed_docs = bool(referenced_doc_ids & removed_doc_ids)
        references_official_docs = bool(referenced_doc_ids & official_doc_ids)
        should_remove = qid in removed_qids or references_removed_docs or (
            is_test_like_qid(qid) and not references_official_docs and not is_official_qid(qid)
        )
        if not should_remove:
            kept_evidence.append(row)

    write_json_array(DOCUMENTS_FILE, kept_documents)
    write_json_array(CHUNKS_FILE, kept_chunks)
    write_json_array(TASKS_FILE, kept_tasks)
    write_json_array(RESULTS_FILE, kept_results)
    write_json_array(EVIDENCE_FILE, kept_evidence)
    answer_stats = rewrite_answer_csv(ANSWER_FILE, removed_qids)

    print(
        json.dumps(
            {
                "removed_doc_count": len(removed_doc_ids),
                "remaining_documents": len(kept_documents),
                "remaining_chunks": len(kept_chunks),
                "removed_result_count": len(results) - len(kept_results),
                "remaining_results": len(kept_results),
                "removed_task_count": len(tasks) - len(kept_tasks),
                "remaining_tasks": len(kept_tasks),
                "removed_evidence_count": len(evidence_rows) - len(kept_evidence),
                "remaining_evidence": len(kept_evidence),
                "removed_qids": sorted(removed_qids),
                **answer_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
