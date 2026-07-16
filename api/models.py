from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TaskMode = Literal["A", "B"]
AnswerFormat = Literal["single", "multi", "judge"]
TaskStatus = Literal["queued", "running", "done", "failed"]
Verdict = Literal["support", "refute", "insufficient"]


class UploadDocumentRequest(BaseModel):
    file_name: str = Field(alias="fileName")
    file_type: Literal["pdf", "txt"] = Field(alias="fileType")
    source_path: str | None = Field(default=None, alias="sourcePath")


class UploadDocumentResponse(BaseModel):
    document_id: str = Field(alias="documentId")
    status: TaskStatus


class StartPreprocessRequest(BaseModel):
    document_id: str = Field(alias="documentId")
    enable_ocr: bool = Field(default=True, alias="enableOcr")
    enable_table_recovery: bool = Field(default=True, alias="enableTableRecovery")


class StartPreprocessResponse(BaseModel):
    task_id: str = Field(alias="taskId")
    status: TaskStatus


class RunQuestionTaskRequest(BaseModel):
    mode: TaskMode
    qid: str
    question: str
    options: list[str]
    answer_format: AnswerFormat = Field(alias="answerFormat")
    doc_ids: list[str] | None = Field(default=None, alias="docIds")


class EvidenceItem(BaseModel):
    doc_id: str = Field(alias="docId")
    page_no: int = Field(alias="pageNo")
    clause_no: str | None = Field(default=None, alias="clauseNo")
    quoted_text: str = Field(alias="quotedText")
    supports_option: list[str] = Field(default_factory=list, alias="supportsOption")
    reasoning: str


class TaskResultResponse(BaseModel):
    qid: str
    answer: str
    evidence: list[EvidenceItem]
    token_usage: dict[str, int] = Field(alias="tokenUsage")
    llm_trace: LLMTrace = Field(alias="llmTrace")
    logs: list[str]


class RunQuestionTaskResponse(BaseModel):
    task_id: str = Field(alias="taskId")
    status: TaskStatus


class StructuredChunk(BaseModel):
    chunk_id: str = Field(alias="chunkId")
    doc_id: str = Field(alias="docId")
    title: str
    domain: str
    page_no: int = Field(alias="pageNo")
    section_path: str = Field(alias="sectionPath")
    chunk_type: str = Field(alias="chunkType")
    clause_no: str | None = Field(default=None, alias="clauseNo")
    chunk_text: str = Field(alias="chunkText")


class DocumentRecord(BaseModel):
    document_id: str = Field(alias="documentId")
    title: str
    file_type: str = Field(alias="fileType")
    source_path: str = Field(alias="sourcePath")
    status: TaskStatus
    chunk_count: int = Field(default=0, alias="chunkCount")


class ReasoningItem(BaseModel):
    option: str
    verdict: Verdict
    reasoning: str


class LLMTrace(BaseModel):
    enabled: bool
    attempted: bool
    used: str
    requested_model: str = Field(alias="requestedModel")
    actual_model: str = Field(alias="actualModel")
    prompt_tokens: int = Field(alias="promptTokens")
    completion_tokens: int = Field(alias="completionTokens")
    total_tokens: int = Field(alias="totalTokens")
    fallback: bool
    fallback_reason: str = Field(alias="fallbackReason")
    error: str | None = None
    missing_settings: list[str] = Field(default_factory=list, alias="missingSettings")


class DeliveryArtifact(BaseModel):
    name: str
    description: str
    path: str
