from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.models import DocumentRecord  # noqa: E402
from api.services.preprocess import preprocess_document  # noqa: E402
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
    TEMP_CHUNKS_JSONL,
    TEMP_DOCS_JSONL,
    normalize_temp_files,
)


def flush_handle(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def process_direct(file_path: Path) -> tuple[ProcessedDocument, list[dict]]:
    relative_parts = file_path.relative_to(ROOT / "public_dataset_a" / "raw").parts
    category = relative_parts[0]
    subcategory = relative_parts[1] if len(relative_parts) > 2 else "root"
    file_type = detect_file_type(file_path)
    title, metadata = build_title(file_path, file_type)
    doc_id = file_path.stem

    record = DocumentRecord(
        documentId=doc_id,
        title=title,
        fileType=file_type,
        sourcePath=str(file_path),
        status="queued",
        chunkCount=0,
    )
    prefer_pdfplumber = category == "research"
    enable_table_recovery = (
        file_type == "pdf"
        and category != "financial_reports"
        and file_path.stat().st_size <= 8 * 1024 * 1024
        and not prefer_pdfplumber
        and not (category == "regulatory" and subcategory == "attachments")
    )

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
    notes = "" if status == "done" else "empty_or_fallback_content"

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
    return processed, chunk_dicts


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    processed_ids = normalize_temp_files()
    all_files = iter_files()
    remaining = [path for path in all_files if path.stem not in processed_ids]

    with TEMP_DOCS_JSONL.open("a", encoding="utf-8") as docs_handle, TEMP_CHUNKS_JSONL.open("a", encoding="utf-8") as chunks_handle:
        for index, file_path in enumerate(remaining, start=1):
            processed_doc, chunk_dicts = process_direct(file_path)
            docs_handle.write(json.dumps(asdict(processed_doc), ensure_ascii=False) + "\n")
            flush_handle(docs_handle)
            for chunk in chunk_dicts:
                chunks_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            flush_handle(chunks_handle)
            processed_ids.add(processed_doc.doc_id)
            print(
                json.dumps(
                    {
                        "batch_progress": f"{index}/{len(remaining)}",
                        "overall_progress": f"{len(processed_ids)}/{len(all_files)}",
                        "doc_id": processed_doc.doc_id,
                        "status": processed_doc.status,
                        "chunks": len(chunk_dicts),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if len(processed_ids) == len(all_files):
        TEMP_DOCS_JSONL.replace(DOCS_JSONL)
        TEMP_CHUNKS_JSONL.replace(CHUNKS_JSONL)
        print(json.dumps({"done": True, "documents": str(DOCS_JSONL), "chunks": str(CHUNKS_JSONL)}, ensure_ascii=False))
    else:
        print(json.dumps({"done": False, "remaining": len(all_files) - len(processed_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
