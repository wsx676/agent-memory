from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_processed_doc_ids() -> set[str]:
    from scripts.preprocess_public_dataset_a import DOCS_JSONL

    doc_ids: set[str] = set()
    if not DOCS_JSONL.exists():
        return doc_ids
    with DOCS_JSONL.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                doc_ids.add(json.loads(line)["doc_id"])
            except Exception:
                continue
    return doc_ids


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    from scripts.preprocess_public_dataset_a import CHUNKS_JSONL, DOCS_JSONL, iter_files, process_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=0, help="最多补处理多少个文件，0 表示不限制。")
    args = parser.parse_args()

    processed_doc_ids = load_processed_doc_ids()
    remaining_files = [path for path in iter_files() if path.stem not in processed_doc_ids]
    if args.max_files > 0:
        remaining_files = remaining_files[: args.max_files]
    print(json.dumps({"remaining": len(remaining_files)}, ensure_ascii=False), flush=True)

    for index, file_path in enumerate(remaining_files, start=1):
        processed_doc, chunk_dicts = process_file(file_path)
        append_jsonl(DOCS_JSONL, [asdict(processed_doc)])
        append_jsonl(CHUNKS_JSONL, chunk_dicts)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(remaining_files)}",
                    "doc_id": processed_doc.doc_id,
                    "status": processed_doc.status,
                    "chunks": processed_doc.chunk_count,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
