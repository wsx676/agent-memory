from __future__ import annotations

import json
import os
import sys
import time
import threading
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET_ROOT = ROOT / "public_dataset_a"
RAW_ROOT = DATASET_ROOT / "raw"
OUTPUT_ROOT = ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
DOCS_JSONL = OUTPUT_ROOT / "documents.jsonl"
CHUNKS_JSONL = OUTPUT_ROOT / "chunks.jsonl"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"


@dataclass
class ProcessedDocument:
    doc_id: str
    title: str
    category: str
    subcategory: str
    file_type: str
    file_path: str
    processing_seconds: float
    chunk_count: int
    page_count: int
    char_count: int
    clause_chunk_count: int
    table_chunk_count: int
    metadata_title: str
    metadata_pub_date: str
    metadata_source: str
    status: str
    notes: str


# #region debug-point D:report
def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict[str, object]) -> None:
    env_path = ROOT / ".dbg" / "pdf-preprocess-gap.env"
    debug_server_url = "http://127.0.0.1:7777/event"
    debug_session_id = "pdf-preprocess-gap"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    debug_session_id = line.split("=", 1)[1].strip()
        payload = {
            "sessionId": debug_session_id,
            "runId": os.environ.get("DEBUG_RUN_ID", "pre-fix"),
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


def iter_files() -> list[Path]:
    return sorted((path for path in RAW_ROOT.rglob("*") if path.is_file()), key=lambda item: (item.suffix.lower(), item.stat().st_size))


def detect_file_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    if suffix in {".html", ".htm"}:
        return "html"
    return "unknown"


def build_title(file_path: Path, file_type: str) -> tuple[str, dict[str, str]]:
    metadata = {"title": "", "pub_date": "", "source": ""}
    if file_type == "html":
        from api.services.preprocess import extract_html_metadata

        metadata = extract_html_metadata(file_path)
        title = metadata["title"] or file_path.stem
        return title, metadata
    return file_path.stem, metadata


def process_file(file_path: Path) -> tuple[ProcessedDocument, list[dict]]:
    from api.models import DocumentRecord
    from api.services.preprocess import preprocess_document

    relative_parts = file_path.relative_to(RAW_ROOT).parts
    category = relative_parts[0]
    subcategory = relative_parts[1] if len(relative_parts) > 2 else "root"
    file_type = detect_file_type(file_path)
    title, metadata = build_title(file_path, file_type)
    doc_id = file_path.stem

    if file_type == "unknown":
        return (
            ProcessedDocument(
                doc_id=doc_id,
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
                status="skipped",
                notes="unsupported_file_type",
            ),
            [],
        )

    record = DocumentRecord(
        documentId=doc_id,
        title=title,
        fileType=file_type,
        sourcePath=str(file_path),
        status="queued",
        chunkCount=0,
    )
    prefer_pdfplumber = category == "research"
    # 年报（financial_reports）放宽表格恢复的大小上限至 64MB。
    # 年报 PDF 普遍 >8MB，原 8MB 限制（连同 `category != "financial_reports"`）
    # 会直接挡掉表格恢复，导致财报类题目的核心数据（资产负债表/利润表）全部丢失。
    # 其余域保持 <8MB 以控制离线处理耗时。
    table_size_limit = 64 * 1024 * 1024 if category == "financial_reports" else 8 * 1024 * 1024
    enable_table_recovery = (
        file_type == "pdf"
        and file_path.stat().st_size <= table_size_limit
        and not prefer_pdfplumber
        and not (category == "regulatory" and subcategory == "attachments")
    )
    # #region debug-point D:process-start
    _debug_report(
        "D",
        "preprocess_public_dataset_a.py:process_file",
        "[DEBUG] process_file start",
        {
            "doc_id": doc_id,
            "file_path": str(file_path),
            "file_type": file_type,
            "category": category,
            "enable_table_recovery": enable_table_recovery,
            "prefer_pdfplumber": prefer_pdfplumber,
        },
    )
    # #endregion
    started_at = time.perf_counter()
    chunks = preprocess_document(
        record,
        domain=category,
        enable_ocr=False,
        enable_table_recovery=enable_table_recovery,
        prefer_pdfplumber=prefer_pdfplumber,
    )
    elapsed = time.perf_counter() - started_at

    chunk_dicts = [chunk.model_dump(by_alias=True) for chunk in chunks]
    page_count = len({item["pageNo"] for item in chunk_dicts})
    char_count = sum(len(item["chunkText"]) for item in chunk_dicts)
    clause_chunk_count = sum(1 for item in chunk_dicts if item["chunkType"] == "clause")
    table_chunk_count = sum(1 for item in chunk_dicts if item["chunkType"] == "table")
    status = "done" if chunk_dicts and not chunk_dicts[0]["chunkText"].startswith("未能解析出正文内容") else "failed"
    notes = ""
    if status == "failed":
        notes = "empty_or_fallback_content"

    processed = ProcessedDocument(
        doc_id=doc_id,
        title=title,
        category=category,
        subcategory=subcategory,
        file_type=file_type,
        file_path=str(file_path),
        processing_seconds=round(elapsed, 4),
        chunk_count=len(chunk_dicts),
        page_count=page_count,
        char_count=char_count,
        clause_chunk_count=clause_chunk_count,
        table_chunk_count=table_chunk_count,
        metadata_title=metadata["title"],
        metadata_pub_date=metadata["pub_date"],
        metadata_source=metadata["source"],
        status=status,
        notes=notes,
    )
    # #region debug-point D:process-end
    _debug_report(
        "D",
        "preprocess_public_dataset_a.py:process_file",
        "[DEBUG] process_file finished",
        {
            "doc_id": doc_id,
            "file_path": str(file_path),
            "status": status,
            "chunk_count": len(chunk_dicts),
            "page_count": page_count,
            "processing_seconds": round(elapsed, 4),
            "notes": notes,
        },
    )
    # #endregion
    return processed, chunk_dicts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing_docs() -> list[ProcessedDocument]:
    if not DOCS_JSONL.exists():
        return []
    docs: list[ProcessedDocument] = []
    with DOCS_JSONL.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    docs.append(ProcessedDocument(**json.loads(line)))
                except json.JSONDecodeError:
                    continue
    return docs


def summarize(docs: list[ProcessedDocument], total_chunks: int) -> dict[str, object]:
    by_type = Counter(doc.file_type for doc in docs)
    by_status = Counter(doc.status for doc in docs)
    by_category = Counter(doc.category for doc in docs)
    return {
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "by_type": dict(by_type),
        "by_status": dict(by_status),
        "by_category": dict(by_category),
        "avg_processing_seconds": round(sum(doc.processing_seconds for doc in docs) / max(1, len(docs)), 4),
        "avg_chunk_count": round(sum(doc.chunk_count for doc in docs) / max(1, len(docs)), 2),
        "documents_jsonl": str(DOCS_JSONL),
        "chunks_jsonl": str(CHUNKS_JSONL),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    processed_docs = read_existing_docs()
    processed_doc_ids = {doc.doc_id for doc in processed_docs}
    file_paths = [path for path in iter_files() if path.stem not in processed_doc_ids]
    total_chunks = 0
    if CHUNKS_JSONL.exists():
        with CHUNKS_JSONL.open("r", encoding="utf-8", errors="ignore") as handle:
            total_chunks = sum(1 for _ in handle)
    progress = {"processed": len(processed_docs), "last_doc": "", "last_status": "", "total": len(processed_docs) + len(file_paths)}
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(5):
            print(
                json.dumps(
                    {
                        "heartbeat": True,
                        "processed": progress["processed"],
                        "total": progress["total"],
                        "last_doc": progress["last_doc"],
                        "last_status": progress["last_status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()

    with DOCS_JSONL.open("a", encoding="utf-8") as docs_handle, CHUNKS_JSONL.open("a", encoding="utf-8") as chunks_handle:
        for index, file_path in enumerate(file_paths, start=1):
            processed_doc, chunk_dicts = process_file(file_path)
            processed_docs.append(processed_doc)
            total_chunks += len(chunk_dicts)
            progress["processed"] = len(processed_docs)
            progress["last_doc"] = processed_doc.doc_id
            progress["last_status"] = processed_doc.status
            docs_handle.write(json.dumps(asdict(processed_doc), ensure_ascii=False) + "\n")
            for chunk in chunk_dicts:
                chunks_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            if progress["processed"] % 25 == 0:
                print(
                    json.dumps(
                        {
                            "progress": f"{progress['processed']}/{progress['total']}",
                            "last_doc": processed_doc.doc_id,
                            "last_status": processed_doc.status,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = summarize(processed_docs, total_chunks)
    stop_event.set()
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
