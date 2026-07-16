from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    DeliveryArtifact,
    DocumentRecord,
    LLMTrace,
    RunQuestionTaskRequest,
    StartPreprocessRequest,
    StructuredChunk,
    TaskResultResponse,
    UploadDocumentRequest,
)
from api.services.formatter import build_evidence_items, choose_answer
from api.services.memory import build_memory_ledger
from api.services.preprocess import create_document_record, preprocess_document
from api.services.qwen_client import answer_with_qwen, get_qwen_config_status
from api.services.retrieval_loop import run_retrieval_loop
from api.state import ANSWER_FILE, CHUNKS_FILE, DOCUMENTS_FILE, EVIDENCE_FILE, RESULTS_FILE, TASKS_FILE, read_json, write_json

app = FastAPI(title="金融长文档智能阅读理解系统 MVP", version="0.1.0")
PREPROCESSED_DIR = Path(__file__).resolve().parent.parent / "validation_outputs" / "public_dataset_a" / "preprocessed"
PREPROCESSED_DOCUMENTS_FILE = PREPROCESSED_DIR / "documents.jsonl"
# 优先加载合并去重后的切片（17K 条、中位数 450 字），回退到原始切片（61K 条碎片）。
# postprocess_chunks.py 产出的 chunks_merged.jsonl 切片质量显著优于 chunks.jsonl。
PREPROCESSED_CHUNKS_FILE = PREPROCESSED_DIR / "chunks_merged.jsonl"
PREPROCESSED_CHUNKS_FALLBACK_FILE = PREPROCESSED_DIR / "chunks.jsonl"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_preprocessed_documents() -> list[DocumentRecord]:
    records = _read_jsonl(PREPROCESSED_DOCUMENTS_FILE)
    return [
        DocumentRecord(
            documentId=item["doc_id"],
            title=item.get("title", item["doc_id"]),
            fileType=item.get("file_type", "txt"),
            sourcePath=item.get("file_path", ""),
            status=item.get("status", "done"),
            chunkCount=item.get("chunk_count", 0),
        )
        for item in records
        if item.get("status", "done") == "done"
    ]


def _load_local_documents() -> list[DocumentRecord]:
    return [DocumentRecord.model_validate(item) for item in read_json(DOCUMENTS_FILE)]


def _merge_documents(local_documents: list[DocumentRecord], preprocessed_documents: list[DocumentRecord]) -> list[DocumentRecord]:
    local_doc_ids = {item.document_id for item in local_documents}
    merged = list(local_documents)
    merged.extend(item for item in preprocessed_documents if item.document_id not in local_doc_ids)
    return merged


def _load_documents() -> list[DocumentRecord]:
    local_documents = _load_local_documents()
    preprocessed_documents = _load_preprocessed_documents()
    if not local_documents:
        return preprocessed_documents
    if not preprocessed_documents:
        return local_documents
    return _merge_documents(local_documents, preprocessed_documents)


def _save_documents(items: list[DocumentRecord]) -> None:
    write_json(DOCUMENTS_FILE, [item.model_dump(by_alias=True) for item in items])


def _load_local_chunks() -> list[dict]:
    return read_json(CHUNKS_FILE)


def _merge_chunks(local_chunks: list[dict], preprocessed_chunks: list[dict]) -> list[dict]:
    local_doc_ids = {str(item.get("docId")) for item in local_chunks if item.get("docId")}
    local_chunk_ids = {str(item.get("chunkId")) for item in local_chunks if item.get("chunkId")}
    merged = [
        item
        for item in preprocessed_chunks
        if str(item.get("docId")) not in local_doc_ids and str(item.get("chunkId")) not in local_chunk_ids
    ]
    merged.extend(local_chunks)
    return merged


def _load_chunks() -> list[dict]:
    local_chunks = _load_local_chunks()
    # 优先加载合并去重产物；缺失时回退到原始切片，避免因文件未生成而空载。
    preprocessed_chunks = _read_jsonl(PREPROCESSED_CHUNKS_FILE)
    if not preprocessed_chunks and PREPROCESSED_CHUNKS_FALLBACK_FILE.exists():
        preprocessed_chunks = _read_jsonl(PREPROCESSED_CHUNKS_FALLBACK_FILE)
    if not local_chunks:
        return preprocessed_chunks
    if not preprocessed_chunks:
        return local_chunks
    return _merge_chunks(local_chunks, preprocessed_chunks)


def _save_chunks(items: list[dict]) -> None:
    write_json(CHUNKS_FILE, items)


@app.get("/api/health")
def health() -> dict[str, object]:
    qwen_status = get_qwen_config_status()
    return {
        "status": "ok",
        "qwenEnabled": qwen_status.enabled,
        "qwenModel": qwen_status.requested_model,
        "qwenMissingSettings": list(qwen_status.missing_settings),
    }


@app.get("/api/dashboard")
def dashboard() -> dict[str, object]:
    documents = _load_documents()
    results = read_json(RESULTS_FILE)
    qwen_status = get_qwen_config_status()
    return {
        "documents": len(documents),
        "processedDocuments": len([item for item in documents if item.status == "done"]),
        "tasks": len(read_json(TASKS_FILE)),
        "results": len(results),
        "deliveryReady": Path(ANSWER_FILE).exists(),
        "qwenEnabled": qwen_status.enabled,
        "qwenModel": qwen_status.requested_model,
        "qwenMissingSettings": list(qwen_status.missing_settings),
    }


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return [item.model_dump(by_alias=True) for item in _load_documents()]


@app.post("/api/documents/upload")
def upload_document(payload: UploadDocumentRequest) -> dict:
    documents = _load_documents()
    record = create_document_record(payload.file_name, payload.file_type, payload.source_path)
    documents.append(record)
    _save_documents(documents)
    return {"documentId": record.document_id, "status": "queued"}


@app.post("/api/preprocess/start")
def start_preprocess(payload: StartPreprocessRequest) -> dict:
    documents = _load_documents()
    matched = next((item for item in documents if item.document_id == payload.document_id), None)
    if matched is None:
        raise HTTPException(status_code=404, detail="document not found")

    task_id = str(uuid.uuid4())
    matched.status = "running"
    chunks = preprocess_document(
        matched,
        "general",
        enable_ocr=payload.enable_ocr,
        enable_table_recovery=payload.enable_table_recovery,
    )
    matched.status = "done"
    matched.chunk_count = len(chunks)
    _save_documents(documents)
    current_chunks = _load_chunks()
    current_chunks.extend([chunk.model_dump(by_alias=True) for chunk in chunks])
    _save_chunks(current_chunks)

    tasks = read_json(TASKS_FILE)
    tasks.append({"taskId": task_id, "type": "preprocess", "documentId": payload.document_id, "status": "done"})
    write_json(TASKS_FILE, tasks)
    return {"taskId": task_id, "status": "done"}


@app.post("/api/tasks/run")
def run_task(payload: RunQuestionTaskRequest) -> dict:
    documents = _load_documents()
    task_id = str(uuid.uuid4())
    all_chunks = _load_chunks()
    if not all_chunks:
        raise HTTPException(status_code=400, detail="no processed chunks available")
    structured_chunks = [StructuredChunk.model_validate(chunk) for chunk in all_chunks]
    chunk_doc_ids = {chunk.doc_id for chunk in structured_chunks}
    available_doc_ids = {item.document_id for item in documents if item.status == "done" and item.document_id in chunk_doc_ids}
    retrieval_result = run_retrieval_loop(
        payload,
        available_doc_ids=available_doc_ids,
        structured_chunks=structured_chunks,
    )
    plan = dict(retrieval_result["plan"])
    doc_ids = list(retrieval_result["doc_ids"])
    candidate_chunks = list(retrieval_result["candidate_chunks"])
    reasoning_results = list(retrieval_result["reasoning_results"])
    evidence_map = dict(retrieval_result["evidence_map"])
    evaluation = dict(retrieval_result["evaluation"])
    fallback_answer = choose_answer(payload.answer_format, reasoning_results)
    answer = fallback_answer
    qwen_status = get_qwen_config_status()
    qwen_result = None
    qwen_error = None
    qwen_attempted = False
    if qwen_status.enabled:
        qwen_attempted = True
        try:
            qwen_result = answer_with_qwen(
                question=payload.question,
                options=payload.options,
                answer_format=payload.answer_format,
                evidence=candidate_chunks,
                reasoning_hints=reasoning_results,
            )
        except Exception as exc:  # noqa: BLE001
            qwen_error = f"{type(exc).__name__}: {exc}"
    if qwen_result is not None:
        answer = qwen_result.answer
    evidence_items = build_evidence_items(reasoning_results, evidence_map)
    ledger = build_memory_ledger(payload.qid, payload.question, evidence_map, reasoning_results)
    if qwen_result is not None:
        token_usage = {
            "promptTokens": qwen_result.prompt_tokens,
            "completionTokens": qwen_result.completion_tokens,
            "totalTokens": qwen_result.total_tokens,
        }
        llm_trace = LLMTrace(
            enabled=qwen_status.enabled,
            attempted=qwen_attempted,
            used="qwen",
            requestedModel=qwen_status.requested_model,
            actualModel=qwen_result.model,
            promptTokens=qwen_result.prompt_tokens,
            completionTokens=qwen_result.completion_tokens,
            totalTokens=qwen_result.total_tokens,
            fallback=False,
            fallbackReason="none",
            error=None,
            missingSettings=list(qwen_status.missing_settings),
        )
    else:
        token_usage = {
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
        }
        llm_trace = LLMTrace(
            enabled=qwen_status.enabled,
            attempted=qwen_attempted,
            used="local_rule",
            requestedModel=qwen_status.requested_model,
            actualModel="none",
            promptTokens=0,
            completionTokens=0,
            totalTokens=0,
            fallback=True,
            fallbackReason="qwen_error" if qwen_error else qwen_status.disable_reason,
            error=qwen_error,
            missingSettings=list(qwen_status.missing_settings),
        )
    result = TaskResultResponse(
        qid=payload.qid,
        answer=answer,
        evidence=evidence_items,
        tokenUsage=token_usage,
        llmTrace=llm_trace,
        logs=[
            f"mode={payload.mode}",
            f"retrieval_plan={plan.get('retrieval_plan', 'unknown')}",
            f"reasoning_plan={plan.get('reasoning_plan', 'unknown')}",
            f"question_type={plan.get('question_type', 'unknown')}",
            f"active_rewrite={plan.get('active_rewrite_name', 'base')}",
            f"candidate_docs={','.join(doc_ids) if doc_ids else 'none'}",
            f"retrieval_quality={evaluation.get('quality_score', 0.0)}",
            f"retrieval_quality_label={evaluation.get('quality_label', 'unknown')}",
            f"retrieval_failures={','.join(evaluation.get('failure_reasons', [])) or 'none'}",
            f"fallback_answer={fallback_answer}",
            f"qwen_enabled={str(qwen_status.enabled).lower()}",
            f"qwen_requested_model={qwen_status.requested_model}",
            f"qwen_missing_settings={','.join(qwen_status.missing_settings) or 'none'}",
            f"llm_attempted={str(qwen_attempted).lower()}",
            f"llm_used={'qwen' if qwen_result is not None else 'local_rule'}",
            f"llm_model={qwen_result.model if qwen_result is not None else 'none'}",
            f"llm_prompt_tokens={token_usage['promptTokens']}",
            f"llm_completion_tokens={token_usage['completionTokens']}",
            f"llm_total_tokens={token_usage['totalTokens']}",
            f"llm_fallback={str(qwen_result is None).lower()}",
            f"llm_fallback_reason={llm_trace.fallback_reason}",
            f"llm_error={qwen_error or 'none'}",
            *list(retrieval_result.get("loop_logs", [])),
        ],
    )

    tasks = read_json(TASKS_FILE)
    tasks.append({"taskId": task_id, "type": "question", "qid": payload.qid, "status": "done"})
    write_json(TASKS_FILE, tasks)

    results = read_json(RESULTS_FILE)
    results.append({"taskId": task_id, **result.model_dump(by_alias=True), "ledger": ledger})
    write_json(RESULTS_FILE, results)

    evidence_store = read_json(EVIDENCE_FILE)
    evidence_store.append(
        {
            "qid": payload.qid,
            "domain": plan.get("domain", "general"),
            "answer": answer,
            "evidence_retrieval": [item.model_dump(by_alias=True) for item in evidence_items],
            "token_usage": token_usage,
            "llm_trace": llm_trace.model_dump(by_alias=True),
        }
    )
    write_json(EVIDENCE_FILE, evidence_store)

    with Path(ANSWER_FILE).open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(
            [payload.qid, answer, token_usage["promptTokens"], token_usage["completionTokens"], token_usage["totalTokens"]]
        )

    return {"taskId": task_id, "status": "done"}


@app.get("/api/tasks/{task_id}")
def get_task_result(task_id: str) -> dict:
    result = next((item for item in read_json(RESULTS_FILE) if item.get("taskId") == task_id), None)
    if result is None:
        raise HTTPException(status_code=404, detail="task not found")
    return result


@app.get("/api/results")
def list_results() -> list[dict]:
    return read_json(RESULTS_FILE)


@app.get("/api/quality")
def quality() -> dict[str, object]:
    return {
        "coverageTarget": ">=80%",
        "lint": "pending",
        "typeCheck": "pending",
        "backendTests": "pending",
        "frontendTests": "pending",
        "ci": "configured",
    }


@app.get("/api/delivery")
def delivery() -> list[dict]:
    artifacts = [
        DeliveryArtifact(name="PRD", description="产品需求文档", path=".trae/documents/金融长文档MVP-产品需求文档.md"),
        DeliveryArtifact(name="TechArch", description="技术架构文档", path=".trae/documents/金融长文档MVP-技术架构文档.md"),
        DeliveryArtifact(name="DeployManual", description="部署手册", path="docs/部署手册.md"),
        DeliveryArtifact(name="DeliveryReport", description="交付报告", path="docs/交付报告.md"),
    ]
    return [item.model_dump() for item in artifacts]
