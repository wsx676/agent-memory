from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import asdict

from scripts.preprocess_public_dataset_a import OUTPUT_ROOT, iter_files, process_file  # noqa: E402
from scripts.rebuild_preprocessed_public_dataset_a import TEMP_CHUNKS_JSONL, TEMP_DOCS_JSONL  # noqa: E402


def read_processed_ids() -> set[str]:
    processed: set[str] = set()
    if not TEMP_DOCS_JSONL.exists():
        return processed
    with TEMP_DOCS_JSONL.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = row.get("doc_id")
            if doc_id:
                processed.add(doc_id)
    return processed


def flush_handle(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def main() -> None:
    processed_ids = read_processed_ids()
    remaining = [path for path in iter_files() if path.stem not in processed_ids]
    if not remaining:
        print(json.dumps({"done": True}, ensure_ascii=False))
        return

    file_path = remaining[0]
    processed_doc, chunks = process_file(file_path)
    with TEMP_DOCS_JSONL.open("a", encoding="utf-8") as docs_handle:
        docs_handle.write(json.dumps(asdict(processed_doc), ensure_ascii=False) + "\n")
        flush_handle(docs_handle)
    with TEMP_CHUNKS_JSONL.open("a", encoding="utf-8") as chunks_handle:
        for chunk in chunks:
            chunks_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        flush_handle(chunks_handle)
    print(
        json.dumps(
            {
                "done": False,
                "doc_id": processed_doc.doc_id,
                "status": processed_doc.status,
                "chunk_count": len(chunks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
