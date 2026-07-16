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

from scripts.preprocess_public_dataset_a import CHUNKS_JSONL, DOCS_JSONL, ROOT, process_file


REPORT_JSON = ROOT / "validation_outputs" / "public_dataset_a" / "refresh_missing_chunks_report.json"


# #region debug-point D:report-helper
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


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0, help="从缺失文档列表的哪个位置开始处理")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个缺失文档，0 表示全部")
    parser.add_argument("--commit-every", type=int, default=0, help="每处理多少个文档就增量落盘一次，0 表示仅在最后落盘")
    args = parser.parse_args()

    docs = read_jsonl(DOCS_JSONL)
    chunks = read_jsonl(CHUNKS_JSONL)
    chunk_doc_ids = {row.get("docId") for row in chunks if row.get("docId")}
    target_docs = [row for row in docs if row.get("chunk_count", 0) > 0 and row.get("doc_id") not in chunk_doc_ids]
    if args.offset > 0:
        target_docs = target_docs[args.offset :]
    if args.limit > 0:
        target_docs = target_docs[: args.limit]
    target_ids = {row["doc_id"] for row in target_docs}
    # #region debug-point D:refresh-start
    _debug_report(
        "D",
        "refresh_missing_chunks_public_dataset_a.py:main:start",
        "[DEBUG] refresh start",
        {
            "docs": len(docs),
            "chunks": len(chunks),
            "target_docs": len(target_docs),
            "offset": args.offset,
            "limit": args.limit,
            "commit_every": args.commit_every,
            "sample_target_doc_ids": [row["doc_id"] for row in target_docs[:10]],
        },
    )
    # #endregion

    print(json.dumps({"targets": len(target_docs)}, ensure_ascii=False))

    if not target_docs:
        return

    new_docs: list[dict] = []
    new_chunks: list[dict] = []
    for index, row in enumerate(target_docs, start=1):
        processed, chunk_dicts = process_file(Path(row["file_path"]))
        new_docs.append(asdict(processed))
        new_chunks.extend(chunk_dicts)
        # #region debug-point D:refresh-progress
        _debug_report(
            "D",
            "refresh_missing_chunks_public_dataset_a.py:main:progress",
            "[DEBUG] refresh processed one missing doc",
            {
                "index": index,
                "target_total": len(target_docs),
                "doc_id": processed.doc_id,
                "status": processed.status,
                "generated_chunks": len(chunk_dicts),
            },
        )
        # #endregion
        if index % 20 == 0 or index == len(target_docs):
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(target_docs)}",
                        "doc_id": row["doc_id"],
                        "generated_chunks": len(chunk_dicts),
                    },
                    ensure_ascii=False,
                )
            )
        if args.commit_every > 0 and index % args.commit_every == 0:
            commit_doc_ids = {item["doc_id"] for item in new_docs}
            docs = [item for item in docs if item.get("doc_id") not in commit_doc_ids]
            docs.extend(new_docs)
            chunks = [item for item in chunks if item.get("docId") not in commit_doc_ids]
            chunks.extend(new_chunks)
            write_jsonl(DOCS_JSONL, docs)
            write_jsonl(CHUNKS_JSONL, chunks)
            # #region debug-point D:refresh-incremental-commit
            _debug_report(
                "D",
                "refresh_missing_chunks_public_dataset_a.py:main:incremental_commit",
                "[DEBUG] refresh incremental commit",
                {
                    "index": index,
                    "commit_every": args.commit_every,
                    "committed_docs": len(commit_doc_ids),
                    "docs_total": len(docs),
                    "chunks_total": len(chunks),
                },
            )
            # #endregion
            new_docs = []
            new_chunks = []

    merged_doc_ids = {item["doc_id"] for item in new_docs}
    merged_docs = [row for row in docs if row.get("doc_id") not in merged_doc_ids]
    merged_docs.extend(new_docs)
    merged_chunk_doc_ids = {row.get("docId") for row in new_chunks if row.get("docId")}
    merged_chunks = [row for row in chunks if row.get("docId") not in merged_chunk_doc_ids]
    merged_chunks.extend(new_chunks)
    merged_doc_ids = {row.get("docId") for row in merged_chunks if row.get("docId")}
    remaining_before_write = sorted(
        row["doc_id"] for row in merged_docs if row.get("chunk_count", 0) > 0 and row.get("doc_id") not in merged_doc_ids
    )
    # #region debug-point D:refresh-before-write
    _debug_report(
        "D",
        "refresh_missing_chunks_public_dataset_a.py:main:before_write",
        "[DEBUG] refresh merged before write",
        {
            "updated_docs": len(new_docs),
            "updated_chunks": len(new_chunks),
            "merged_docs": len(merged_docs),
            "merged_chunks": len(merged_chunks),
            "remaining_before_write": len(remaining_before_write),
            "sample_remaining_before_write": remaining_before_write[:10],
        },
    )
    # #endregion

    write_jsonl(DOCS_JSONL, merged_docs)
    write_jsonl(CHUNKS_JSONL, merged_chunks)

    refreshed_chunks = read_jsonl(CHUNKS_JSONL)
    refreshed_doc_ids = {row.get("docId") for row in refreshed_chunks if row.get("docId")}
    remaining_targets = sorted(doc_id for doc_id in target_ids if doc_id not in refreshed_doc_ids)
    # #region debug-point D:refresh-after-write
    _debug_report(
        "D",
        "refresh_missing_chunks_public_dataset_a.py:main:after_write",
        "[DEBUG] refresh reread after write",
        {
            "refreshed_chunks": len(refreshed_chunks),
            "remaining_missing_chunk_docs": len(remaining_targets),
            "sample_remaining_doc_ids": remaining_targets[:10],
        },
    )
    # #endregion
    report = {
        "updated_docs": len(target_docs),
        "updated_chunks": len([row for row in refreshed_chunks if row.get("docId") in target_ids]),
        "remaining_missing_chunk_docs": len(remaining_targets),
        "remaining_doc_ids": remaining_targets[:10],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
