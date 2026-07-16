from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RAW_ROOT = ROOT / "public_dataset_a" / "raw"
PREPROCESS_ROOT = ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
DOCS_JSONL = PREPROCESS_ROOT / "documents.jsonl"
CHUNKS_JSONL = PREPROCESS_ROOT / "chunks.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def raw_doc_ids() -> set[str]:
    return {path.stem for path in RAW_ROOT.rglob("*") if path.is_file()}


def dedupe_docs(rows: list[dict]) -> list[dict]:
    latest_by_doc_id: dict[str, dict] = {}
    for row in rows:
        doc_id = row.get("doc_id")
        if doc_id:
            latest_by_doc_id[doc_id] = row
    return list(latest_by_doc_id.values())


def dedupe_chunks(rows: list[dict], valid_doc_ids: set[str]) -> list[dict]:
    latest_by_chunk_id: dict[str, dict] = {}
    for row in rows:
        doc_id = row.get("docId")
        chunk_id = row.get("chunkId")
        if not doc_id or not chunk_id or doc_id not in valid_doc_ids:
            continue
        latest_by_chunk_id[chunk_id] = row
    return list(latest_by_chunk_id.values())


def main() -> None:
    PREPROCESS_ROOT.mkdir(parents=True, exist_ok=True)
    valid_doc_ids = raw_doc_ids()
    deduped_docs = dedupe_docs(read_jsonl(DOCS_JSONL))
    deduped_chunks = dedupe_chunks(read_jsonl(CHUNKS_JSONL), valid_doc_ids)
    write_jsonl(DOCS_JSONL, deduped_docs)
    write_jsonl(CHUNKS_JSONL, deduped_chunks)
    print(
        json.dumps(
            {
                "documents": len(deduped_docs),
                "chunks": len(deduped_chunks),
                "missing_documents": len(valid_doc_ids - {row["doc_id"] for row in deduped_docs}),
                "missing_chunk_documents": len(valid_doc_ids - {row["docId"] for row in deduped_chunks}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
