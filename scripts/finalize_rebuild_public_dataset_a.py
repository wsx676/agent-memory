from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_public_dataset_a import (  # noqa: E402
    OUTPUT_ROOT,
    ProcessedDocument,
    build_title,
    detect_file_type,
    iter_files,
)
from scripts.rebuild_preprocessed_public_dataset_a import (  # noqa: E402
    CHUNKS_JSONL,
    DOCS_JSONL,
    RAW_ROOT,
    TEMP_CHUNKS_JSONL,
    TEMP_DOCS_JSONL,
)

SINGLE_RUN_ROOT = OUTPUT_ROOT / "_single_runs"


# #region debug-point C:report-helper
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
        doc_id = row.get("doc_id")
        if doc_id:
            deduped_docs[doc_id] = row
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


def failed_document(file_path: Path, note: str) -> dict:
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
    ).__dict__


def process_one(file_path: Path, timeout_seconds: int) -> tuple[dict, list[dict]]:
    SINGLE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = SINGLE_RUN_ROOT / f"{file_path.stem}.json"
    if output_path.exists():
        output_path.unlink()

    command = [
        sys.executable,
        "-m",
        "scripts.process_single_public_dataset_a",
        "--file",
        str(file_path),
        "--output",
        str(output_path),
    ]
    # #region debug-point C:subprocess-start
    _debug_report(
        "C",
        "finalize_rebuild_public_dataset_a.py:process_one:start",
        "[DEBUG] finalize subprocess start",
        {"file": str(file_path), "output": str(output_path), "timeout_seconds": timeout_seconds},
    )
    # #endregion
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # #region debug-point C:subprocess-timeout
        _debug_report(
            "C",
            "finalize_rebuild_public_dataset_a.py:process_one:timeout",
            "[DEBUG] finalize subprocess timeout",
            {"file": str(file_path), "timeout_seconds": timeout_seconds},
        )
        # #endregion
        return failed_document(file_path, f"subprocess_timeout_{timeout_seconds}s"), []

    # #region debug-point C:subprocess-returned
    _debug_report(
        "C",
        "finalize_rebuild_public_dataset_a.py:process_one:returned",
        "[DEBUG] finalize subprocess returned",
        {
            "file": str(file_path),
            "returncode": completed.returncode,
            "stdout_len": len(completed.stdout or ""),
            "stderr_len": len(completed.stderr or ""),
            "output_exists": output_path.exists(),
        },
    )
    # #endregion
    if completed.returncode != 0:
        return failed_document(file_path, f"subprocess_exit_{completed.returncode}"), []
    if not output_path.exists():
        # #region debug-point C:subprocess-missing-output
        _debug_report(
            "C",
            "finalize_rebuild_public_dataset_a.py:process_one:missing_output",
            "[DEBUG] finalize subprocess missing output",
            {"file": str(file_path), "returncode": completed.returncode},
        )
        # #endregion
        return failed_document(file_path, "subprocess_missing_output"), []

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        processed_doc = payload["processed_doc"]
        chunks = payload["chunks"]
    except Exception as exc:
        return failed_document(file_path, f"subprocess_invalid_output_{type(exc).__name__}"), []
    finally:
        if output_path.exists():
            output_path.unlink()

    return processed_doc, chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个剩余文件，0 表示不限制。")
    parser.add_argument("--timeout-seconds", type=int, default=240, help="单文件子进程超时时间。")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    processed_doc_ids = normalize_temp_files()
    file_paths = [path for path in iter_files() if path.stem not in processed_doc_ids]
    if args.limit > 0:
        file_paths = file_paths[: args.limit]
    total_target = len(iter_files())
    # #region debug-point C:finalize-start
    _debug_report(
        "C",
        "finalize_rebuild_public_dataset_a.py:main:start",
        "[DEBUG] finalize rebuild start",
        {
            "already_processed": len(processed_doc_ids),
            "pending_files": len(file_paths),
            "total_target": total_target,
            "timeout_seconds": args.timeout_seconds,
        },
    )
    # #endregion

    with TEMP_DOCS_JSONL.open("a", encoding="utf-8") as docs_handle, TEMP_CHUNKS_JSONL.open("a", encoding="utf-8") as chunks_handle:
        for index, file_path in enumerate(file_paths, start=1):
            processed_doc, chunks = process_one(file_path, args.timeout_seconds)
            write_jsonl_row(docs_handle, processed_doc)
            for chunk in chunks:
                write_jsonl_row(chunks_handle, chunk)
            flush_handles(docs_handle, chunks_handle)
            processed_doc_ids.add(processed_doc["doc_id"])
            # #region debug-point C:finalize-progress
            _debug_report(
                "C",
                "finalize_rebuild_public_dataset_a.py:main:progress",
                "[DEBUG] finalize rebuild processed one file",
                {
                    "index": index,
                    "pending_total": len(file_paths),
                    "overall_processed": len(processed_doc_ids),
                    "doc_id": processed_doc["doc_id"],
                    "status": processed_doc["status"],
                    "chunks": len(chunks),
                },
            )
            # #endregion
            print(
                json.dumps(
                    {
                        "batch_progress": f"{index}/{len(file_paths)}",
                        "overall_progress": f"{len(processed_doc_ids)}/{total_target}",
                        "doc_id": processed_doc["doc_id"],
                        "status": processed_doc["status"],
                        "chunks": len(chunks),
                        "notes": processed_doc["notes"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if len(processed_doc_ids) == total_target:
        # #region debug-point C:finalize-replace
        _debug_report(
            "C",
            "finalize_rebuild_public_dataset_a.py:main:replace",
            "[DEBUG] finalize rebuild replacing official files",
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
        # #region debug-point C:finalize-incomplete
        _debug_report(
            "C",
            "finalize_rebuild_public_dataset_a.py:main:incomplete",
            "[DEBUG] finalize rebuild incomplete",
            {"processed": len(processed_doc_ids), "total_target": total_target, "remaining": total_target - len(processed_doc_ids)},
        )
        # #endregion
        print(json.dumps({"done": False, "remaining": total_target - len(processed_doc_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
