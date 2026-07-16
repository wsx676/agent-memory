from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_public_dataset_a import (  # noqa: E402
    CHUNKS_JSONL,
    DOCS_JSONL,
    OUTPUT_ROOT,
    ProcessedDocument,
    build_title,
    detect_file_type,
    iter_files,
    process_file,
)

RAW_ROOT = ROOT / "public_dataset_a" / "raw"
TEMP_DOCS_JSONL = OUTPUT_ROOT / "documents.rebuild.jsonl"
TEMP_CHUNKS_JSONL = OUTPUT_ROOT / "chunks.rebuild.jsonl"


# #region debug-point B:report-helper
def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict[str, object]) -> None:
    env_path = ROOT / ".dbg" / "preprocess-chunk-gap.env"
    debug_server_url = "http://127.0.0.1:7777/event"
    debug_session_id = "preprocess-chunk-gap"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    debug_session_id = line.split("=", 1)[1].strip()
        payload = {
            "sessionId": debug_session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                debug_server_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass


# #endregion


def write_jsonl_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_handles(*handles) -> None:
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())


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


def normalize_temp_files() -> set[str]:
    doc_rows = read_jsonl(TEMP_DOCS_JSONL)
    deduped_docs: dict[str, dict] = {}
    for row in doc_rows:
        deduped_docs[row["doc_id"]] = row
    if deduped_docs:
        with TEMP_DOCS_JSONL.open("w", encoding="utf-8") as handle:
            for row in deduped_docs.values():
                write_jsonl_row(handle, row)

    valid_doc_ids = set(deduped_docs.keys())
    chunk_rows = read_jsonl(TEMP_CHUNKS_JSONL)
    seen_chunk_ids: set[str] = set()
    with TEMP_CHUNKS_JSONL.open("w", encoding="utf-8") as handle:
        for row in chunk_rows:
            doc_id = row.get("docId")
            chunk_id = row.get("chunkId")
            if doc_id not in valid_doc_ids or not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            write_jsonl_row(handle, row)
    return valid_doc_ids


def failed_document(file_path: Path, note: str) -> ProcessedDocument:
    relative_parts = file_path.relative_to(RAW_ROOT).parts
    category = relative_parts[0]
    subcategory = relative_parts[1] if len(relative_parts) > 2 else "root"
    file_type = detect_file_type(file_path)
    title, metadata = build_title(file_path, file_type)
    return ProcessedDocument(
        doc_id=file_path.stem,
        title=title,
        category=category,
        subcategory=subcategory,
        file_type=file_type,
        file_path=str(file_path),
        processing_seconds=0.0,
        chunk_count=0,
        page_count=0,
        char_count=0,
        clause_chunk_count=0,
        table_chunk_count=0,
        metadata_title=metadata["title"],
        metadata_pub_date=metadata["pub_date"],
        metadata_source=metadata["source"],
        status="failed",
        notes=note,
    )


def process_with_guard(file_path: Path) -> tuple[ProcessedDocument, list[dict]]:
    try:
        return process_file(file_path)
    except Exception as exc:
        return failed_document(file_path, f"process_exception_{type(exc).__name__}"), []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="仅重建前 N 个文件，0 表示全量。")
    parser.add_argument("--reset", action="store_true", help="忽略已有临时重建结果并从头开始。")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.reset:
        processed_doc_ids: set[str] = set()
        docs_mode = "w"
        chunks_mode = "w"
    else:
        processed_doc_ids = normalize_temp_files()
        docs_mode = "a" if TEMP_DOCS_JSONL.exists() else "w"
        chunks_mode = "a" if TEMP_CHUNKS_JSONL.exists() else "w"

    file_paths = [path for path in iter_files() if path.stem not in processed_doc_ids]
    if args.limit > 0:
        file_paths = file_paths[: args.limit]
    total_target = len(iter_files())
    # #region debug-point B:rebuild-start
    _debug_report(
        "B",
        "rebuild_preprocessed_public_dataset_a.py:main:start",
        "[DEBUG] rebuild start",
        {
            "reset": args.reset,
            "already_processed": len(processed_doc_ids),
            "pending_files": len(file_paths),
            "total_target": total_target,
        },
    )
    # #endregion
    with TEMP_DOCS_JSONL.open(docs_mode, encoding="utf-8") as docs_handle, TEMP_CHUNKS_JSONL.open(chunks_mode, encoding="utf-8") as chunks_handle:
        for index, file_path in enumerate(file_paths, start=1):
            processed_doc, chunks = process_with_guard(file_path)
            write_jsonl_row(docs_handle, asdict(processed_doc))
            for chunk in chunks:
                write_jsonl_row(chunks_handle, chunk)
            flush_handles(docs_handle, chunks_handle)
            processed_doc_ids.add(processed_doc.doc_id)
            # #region debug-point B:rebuild-progress
            _debug_report(
                "B",
                "rebuild_preprocessed_public_dataset_a.py:main:progress",
                "[DEBUG] rebuild processed one file",
                {
                    "index": index,
                    "pending_total": len(file_paths),
                    "overall_processed": len(processed_doc_ids),
                    "doc_id": processed_doc.doc_id,
                    "status": processed_doc.status,
                    "chunks": len(chunks),
                },
            )
            # #endregion
            if index == 1 or index % 25 == 0 or index == len(file_paths):
                print(
                    json.dumps(
                        {
                            "batch_progress": f"{index}/{len(file_paths)}",
                            "overall_progress": f"{len(processed_doc_ids)}/{total_target}",
                            "doc_id": processed_doc.doc_id,
                            "status": processed_doc.status,
                            "chunks": len(chunks),
                            "notes": processed_doc.notes,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if len(processed_doc_ids) == total_target:
        # #region debug-point B:rebuild-finalize
        _debug_report(
            "B",
            "rebuild_preprocessed_public_dataset_a.py:main:finalize",
            "[DEBUG] rebuild replacing official files",
            {
                "docs_temp": str(TEMP_DOCS_JSONL),
                "chunks_temp": str(TEMP_CHUNKS_JSONL),
                "docs_final": str(DOCS_JSONL),
                "chunks_final": str(CHUNKS_JSONL),
            },
        )
        # #endregion
        TEMP_DOCS_JSONL.replace(DOCS_JSONL)
        TEMP_CHUNKS_JSONL.replace(CHUNKS_JSONL)
        print(json.dumps({"done": True, "documents": str(DOCS_JSONL), "chunks": str(CHUNKS_JSONL)}, ensure_ascii=False))
    else:
        # #region debug-point B:rebuild-incomplete
        _debug_report(
            "B",
            "rebuild_preprocessed_public_dataset_a.py:main:incomplete",
            "[DEBUG] rebuild incomplete",
            {"processed": len(processed_doc_ids), "total_target": total_target, "remaining": total_target - len(processed_doc_ids)},
        )
        # #endregion
        print(json.dumps({"done": False, "remaining": total_target - len(processed_doc_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
