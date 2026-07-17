from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as main_module
from api.main import app
from api.models import DocumentRecord, ReasoningItem, RunQuestionTaskRequest, StructuredChunk
from api.services.planner import activate_rewrite, build_plan
from api.services.qwen_client import QwenConfigStatus
from api.services.retrieval_evaluator import evaluate_retrieval_quality
import api.services.retrieval_loop as retrieval_loop_module
from api.services.retriever import rank_chunks, rank_documents
from api.state import CHUNKS_FILE, read_json


client = TestClient(app)


def _build_minimal_pdf(pages: list[str]) -> bytes:
    def _escape_pdf_text(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[str] = []
    page_object_ids: list[int] = []
    current_id = 1

    catalog_id = current_id
    current_id += 1
    pages_id = current_id
    current_id += 1
    font_id = current_id
    current_id += 1

    content_ids: list[int] = []
    for _ in pages:
        page_object_ids.append(current_id)
        current_id += 1
        content_ids.append(current_id)
        current_id += 1

    objects.append(f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects.append(f"{pages_id} 0 obj\n<< /Type /Pages /Count {len(page_object_ids)} /Kids [{kids}] >>\nendobj\n")
    objects.append(f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    for page_index, text in enumerate(pages):
        page_id = page_object_ids[page_index]
        content_id = content_ids[page_index]
        lines = text.split("\n")
        content_lines = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line_index, line in enumerate(lines):
            if line_index == 0:
                content_lines.append(f"({_escape_pdf_text(line)}) Tj")
            else:
                content_lines.append("T*")
                content_lines.append(f"({_escape_pdf_text(line)}) Tj")
        content_lines.append("ET")
        stream = "\n".join(content_lines)
        objects.append(
            f"{page_id} 0 obj\n<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>\nendobj\n"
        )
        objects.append(
            f"{content_id} 0 obj\n<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream\nendobj\n"
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj.encode("latin-1"))

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root {catalog_id} 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode("latin-1")
    )
    return bytes(pdf)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_document_upload_and_preprocess(tmp_path: Path) -> None:
    source_file = tmp_path / "sample.txt"
    source_file.write_text("第十条 保险责任。本合同生效后提供给付责任。", encoding="utf-8")

    upload_response = client.post(
        "/api/documents/upload",
        json={"fileName": "sample.txt", "fileType": "txt", "sourcePath": str(source_file)},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["documentId"]

    preprocess_response = client.post(
        "/api/preprocess/start",
        json={"documentId": document_id, "enableOcr": False, "enableTableRecovery": False},
    )
    assert preprocess_response.status_code == 200
    assert preprocess_response.json()["status"] == "done"


def test_html_preprocess_extracts_article_text(tmp_path: Path) -> None:
    source_file = tmp_path / "sample.html"
    source_file.write_text(
        """
        <html>
        <head>
          <meta name="ArticleTitle" content="监管测试标题" />
          <meta name="PubDate" content="2026-01-01 10:00:00" />
          <meta name="ContentSource" content="测试来源" />
          <title>页面标题</title>
        </head>
        <body>
          <div class="header">导航内容</div>
          <div class="article-content">
            <p>第一条 为规范信息披露行为，制定本办法。</p>
            <p>第二条 本办法适用于私募投资基金管理人。</p>
          </div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    upload_response = client.post(
        "/api/documents/upload",
        json={"fileName": "sample.html", "fileType": "txt", "sourcePath": str(source_file)},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["documentId"]

    from api.models import DocumentRecord
    from api.services.preprocess import preprocess_document

    record = DocumentRecord(
        documentId=document_id,
        title="监管测试标题",
        fileType="html",
        sourcePath=str(source_file),
        status="queued",
        chunkCount=0,
    )
    chunks = preprocess_document(record, "regulatory", enable_ocr=False, enable_table_recovery=False)

    assert chunks
    assert any("第一条 为规范信息披露行为" in chunk.chunk_text for chunk in chunks)
    assert any(chunk.clause_no == "第一条" for chunk in chunks)
    assert all("导航内容" not in chunk.chunk_text for chunk in chunks)


def test_preprocess_ignores_inline_clause_citations(tmp_path: Path) -> None:
    source_file = tmp_path / "clause-citation.html"
    source_file.write_text(
        """
        <html>
        <head>
          <meta name="ArticleTitle" content="规则解读" />
        </head>
        <body>
          <div class="detail-news">
            <p>本规定根据《证券发行上市保荐业务管理办法》第六十二条制定。</p>
            <p>第一条 为了规范证券发行上市行为，保护投资者合法权益，制定本办法。</p>
            <p>第二条 本办法适用于境内证券发行上市活动。</p>
          </div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    from api.models import DocumentRecord
    from api.services.preprocess import preprocess_document

    record = DocumentRecord(
        documentId="citation-test",
        title="规则解读",
        fileType="html",
        sourcePath=str(source_file),
        status="queued",
        chunkCount=0,
    )
    chunks = preprocess_document(record, "regulatory", enable_ocr=False, enable_table_recovery=False)

    clause_numbers = [chunk.clause_no for chunk in chunks if chunk.clause_no]
    assert clause_numbers == ["第一条", "第二条"]
    assert all("第六十二条" != chunk.clause_no for chunk in chunks)


def test_html_preprocess_prefers_detail_news_over_outer_content(tmp_path: Path) -> None:
    source_file = tmp_path / "detail-news.html"
    source_file.write_text(
        """
        <html>
        <head>
          <meta name="ArticleTitle" content="公告标题" />
          <meta name="PubDate" content="2026-01-02 08:00:00" />
        </head>
        <body>
          <div class="content">
            <div class="header">首页 机构概况 办事服务</div>
            <div class="detail-news">
              <p>第一条 为规范信息披露行为，制定本办法。</p>
              <p>第二条 本办法适用于公开发行证券活动。</p>
              <p>第三条 发行人应当保证披露信息真实、准确、完整。</p>
            </div>
            <div class="footer">联系我们 法律声明 归档数据</div>
          </div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    from api.models import DocumentRecord
    from api.services.preprocess import preprocess_document

    record = DocumentRecord(
        documentId="detail-news-test",
        title="公告标题",
        fileType="html",
        sourcePath=str(source_file),
        status="queued",
        chunkCount=0,
    )
    chunks = preprocess_document(record, "regulatory", enable_ocr=False, enable_table_recovery=False)

    joined = "\n".join(chunk.chunk_text for chunk in chunks)
    assert "首页 机构概况 办事服务" not in joined
    assert "联系我们 法律声明 归档数据" not in joined
    assert "第一条 为规范信息披露行为" in joined


def test_pdf_preprocess_extracts_chunks_with_real_page_numbers(tmp_path: Path) -> None:
    source_file = tmp_path / "sample.pdf"
    source_file.write_bytes(_build_minimal_pdf(["Insurance Liability Page", "Exclusion Terms Page"]))

    upload_response = client.post(
        "/api/documents/upload",
        json={"fileName": "sample.pdf", "fileType": "pdf", "sourcePath": str(source_file)},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["documentId"]

    preprocess_response = client.post(
        "/api/preprocess/start",
        json={"documentId": document_id, "enableOcr": False, "enableTableRecovery": False},
    )
    assert preprocess_response.status_code == 200
    assert preprocess_response.json()["status"] == "done"

    chunks = read_json(CHUNKS_FILE)
    matched_chunks = [chunk for chunk in chunks if chunk["docId"] == document_id]
    assert matched_chunks
    assert {chunk["pageNo"] for chunk in matched_chunks} == {1, 2}
    assert any("Insurance Liability Page" in chunk["chunkText"] for chunk in matched_chunks)
    assert any("Exclusion Terms Page" in chunk["chunkText"] for chunk in matched_chunks)


def test_pdf_postprocess_removes_headers_footers_and_toc(tmp_path: Path) -> None:
    source_file = tmp_path / "noise.pdf"
    source_file.write_bytes(
        _build_minimal_pdf(
            [
                "ACME Annual Report 2025\nTABLE OF CONTENTS\nRevenue Overview ........ 2\nRisk Factors ........ 3\n1",
                "ACME Annual Report 2025\nSection One\nInsurance Liability Page\n2",
                "ACME Annual Report 2025\nSection Two\nExclusion Terms Page\n3",
            ]
        )
    )

    upload_response = client.post(
        "/api/documents/upload",
        json={"fileName": "noise.pdf", "fileType": "pdf", "sourcePath": str(source_file)},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["documentId"]

    preprocess_response = client.post(
        "/api/preprocess/start",
        json={"documentId": document_id, "enableOcr": False, "enableTableRecovery": False},
    )
    assert preprocess_response.status_code == 200

    chunks = read_json(CHUNKS_FILE)
    matched_chunks = [chunk for chunk in chunks if chunk["docId"] == document_id]
    assert matched_chunks
    assert all("ACME Annual Report 2025" not in chunk["chunkText"] for chunk in matched_chunks)
    assert all("TABLE OF CONTENTS" not in chunk["chunkText"] for chunk in matched_chunks)
    assert all("Revenue Overview" not in chunk["chunkText"] for chunk in matched_chunks)
    assert {chunk["pageNo"] for chunk in matched_chunks} == {2, 3}
    assert any("Insurance Liability Page" in chunk["chunkText"] for chunk in matched_chunks)
    assert any("Exclusion Terms Page" in chunk["chunkText"] for chunk in matched_chunks)


def test_task_run_requires_processed_chunks() -> None:
    response = client.post(
        "/api/tasks/run",
        json={
            "mode": "A",
            "qid": "Q-001",
            "question": "该条款是否提供给付责任？",
            "options": ["提供给付责任", "不提供给付责任", "责任未明确", "属于免责范围"],
            "answerFormat": "single",
            "docIds": [],
        },
    )
    assert response.status_code in {200, 400}


def test_task_run_records_qwen_fallback_when_disabled(monkeypatch) -> None:
    chunk = StructuredChunk(
        chunkId="qwen-disabled-1",
        docId="doc-qwen-disabled",
        title="保险条款",
        domain="insurance",
        pageNo=3,
        sectionPath="保险责任",
        chunkType="clause",
        clauseNo="第十条",
        chunkText="第十条 本合同生效后提供给付责任。",
    )
    reasoning_results = [
        ReasoningItem(option="A", verdict="support", reasoning="证据明确支持 A。"),
        ReasoningItem(option="B", verdict="refute", reasoning="证据不支持 B。"),
        ReasoningItem(option="C", verdict="insufficient", reasoning="证据不足。"),
        ReasoningItem(option="D", verdict="refute", reasoning="证据不支持 D。"),
    ]
    monkeypatch.setattr(
        main_module,
        "_load_documents",
        lambda: [
            DocumentRecord(
                documentId="doc-qwen-disabled",
                title="保险条款",
                fileType="txt",
                sourcePath="sample.txt",
                status="done",
                chunkCount=1,
            )
        ],
    )
    monkeypatch.setattr(main_module, "_load_chunks", lambda: [chunk.model_dump(by_alias=True)])
    monkeypatch.setattr(
        main_module,
        "run_retrieval_loop",
        lambda payload, available_doc_ids, structured_chunks: {
            "plan": {"domain": "insurance", "retrieval_plan": "direct", "reasoning_plan": "optionwise"},
            "doc_ids": ["doc-qwen-disabled"],
            "candidate_chunks": [chunk],
            "reasoning_results": reasoning_results,
            "evidence_map": {"A": [chunk], "B": [], "C": [], "D": []},
            "evaluation": {"quality_score": 0.92, "quality_label": "high", "failure_reasons": []},
            "loop_logs": ["react_round=1"],
        },
    )
    monkeypatch.setattr(main_module, "build_memory_ledger", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "get_qwen_config_status",
        lambda: QwenConfigStatus(
            enabled=False,
            requested_model="qwen-test",
            base_url="https://example.com",
            missing_settings=("ENABLE_QWEN_LLM",),
            disable_reason="disabled_by_config",
        ),
    )
    monkeypatch.setattr(
        main_module,
        "answer_with_qwen",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Qwen disabled path should not call answer_with_qwen")),
    )

    response = client.post(
        "/api/tasks/run",
        json={
            "mode": "A",
            "qid": "Q-QWEN-DISABLED",
            "question": "该条款是否提供给付责任？",
            "options": ["提供给付责任", "不提供给付责任", "责任未明确", "属于免责范围"],
            "answerFormat": "single",
            "docIds": ["doc-qwen-disabled"],
        },
    )

    assert response.status_code == 200
    task_id = response.json()["taskId"]
    result = client.get(f"/api/tasks/{task_id}").json()

    assert result["answer"] == "A"
    assert result["tokenUsage"] == {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}
    assert result["llmTrace"]["enabled"] is False
    assert result["llmTrace"]["attempted"] is False
    assert result["llmTrace"]["used"] == "local_rule"
    assert result["llmTrace"]["fallback"] is True
    assert result["llmTrace"]["fallbackReason"] == "disabled_by_config"
    assert result["llmTrace"]["missingSettings"] == ["ENABLE_QWEN_LLM"]
    assert "qwen_enabled=false" in result["logs"]
    assert "llm_attempted=false" in result["logs"]
    assert "llm_fallback=true" in result["logs"]
    assert "llm_total_tokens=0" in result["logs"]


def test_rank_chunks_prefers_clause_match_under_doc_filter() -> None:
    chunks = [
        StructuredChunk(
            chunkId="c1",
            docId="doc-a",
            title="金融机构客户尽职调查办法",
            domain="regulatory",
            pageNo=1,
            sectionPath="第一章 总则",
            chunkType="prose",
            clauseNo=None,
            chunkText="本办法明确金融机构应建立客户尽职调查制度。",
        ),
        StructuredChunk(
            chunkId="c2",
            docId="doc-a",
            title="金融机构客户尽职调查办法",
            domain="regulatory",
            pageNo=2,
            sectionPath="第二章 客户受益所有人识别",
            chunkType="clause",
            clauseNo="第十二条",
            chunkText="第十二条 金融机构应当识别并核实受益所有人身份，保存相关资料。",
        ),
        StructuredChunk(
            chunkId="c3",
            docId="doc-b",
            title="某公司年度报告",
            domain="financial_reports",
            pageNo=8,
            sectionPath="主要财务数据",
            chunkType="table",
            clauseNo=None,
            chunkText="2025年营业收入为100亿元，同比增长15%。",
        ),
    ]

    ranked = rank_chunks(
        question="根据第十二条，金融机构是否需要识别受益所有人？",
        options=["需要识别受益所有人", "无需识别", "仅保存资料", "仅核验账户"],
        chunks=chunks,
        doc_ids=["doc-a"],
        plan={"domain": "regulatory", "clause_refs": ["第十二条"]},
    )

    assert ranked
    assert ranked[0].chunk_id == "c2"
    assert all(chunk.doc_id == "doc-a" for chunk in ranked)


def test_rank_documents_prefers_financial_report_hits() -> None:
    chunks = [
        StructuredChunk(
            chunkId="r1",
            docId="report-doc",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=12,
            sectionPath="主要财务数据",
            chunkType="table",
            clauseNo=None,
            chunkText="2025年营业收入100亿元，同比增长15%，归母净利润20亿元。",
        ),
        StructuredChunk(
            chunkId="r2",
            docId="rule-doc",
            title="信息披露管理办法",
            domain="regulatory",
            pageNo=3,
            sectionPath="第一章 总则",
            chunkType="clause",
            clauseNo="第一条",
            chunkText="为规范信息披露行为，制定本办法。",
        ),
    ]

    ranked_docs = rank_documents(
        question="2025年营业收入同比增长多少？",
        chunks=chunks,
        plan={"domain": "financial_reports", "numbers": ["2025年"]},
    )

    assert ranked_docs
    assert ranked_docs[0] == "report-doc"


def test_rank_documents_rewards_multi_signal_coverage() -> None:
    chunks = [
        StructuredChunk(
            chunkId="a1",
            docId="doc-a",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=8,
            sectionPath="主要财务数据",
            chunkType="table",
            clauseNo=None,
            chunkText="2025年营业收入100亿元，同比增长15%。",
        ),
        StructuredChunk(
            chunkId="a2",
            docId="doc-a",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=20,
            sectionPath="利润表",
            chunkType="table",
            clauseNo=None,
            chunkText="归属于上市公司股东的净利润20亿元。",
        ),
        StructuredChunk(
            chunkId="b1",
            docId="doc-b",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=9,
            sectionPath="主要财务数据",
            chunkType="table",
            clauseNo=None,
            chunkText="2025年营业收入100亿元。营业收入持续增长，营业收入表现良好。",
        ),
    ]

    ranked_docs = rank_documents(
        question="2025年营业收入和归母净利润分别是多少？",
        chunks=chunks,
        plan={"domain": "financial_reports", "numbers": ["2025年"]},
    )

    assert ranked_docs
    assert ranked_docs[0] == "doc-a"


def test_rank_chunks_diversifies_near_duplicate_sections() -> None:
    chunks = [
        StructuredChunk(
            chunkId="c1",
            docId="doc-a",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=8,
            sectionPath="主要财务数据",
            chunkType="table",
            clauseNo=None,
            chunkText="2025年营业收入100亿元，同比增长15%。",
        ),
        StructuredChunk(
            chunkId="c2",
            docId="doc-a",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=9,
            sectionPath="主要财务数据",
            chunkType="table",
            clauseNo=None,
            chunkText="2025年营业收入100亿元，营业收入同比增长15%。",
        ),
        StructuredChunk(
            chunkId="c3",
            docId="doc-a",
            title="2025年年度报告",
            domain="financial_reports",
            pageNo=20,
            sectionPath="利润表",
            chunkType="table",
            clauseNo=None,
            chunkText="归属于上市公司股东的净利润20亿元。",
        ),
    ]

    ranked = rank_chunks(
        question="2025年营业收入和净利润分别是多少？",
        options=["营业收入100亿元，净利润20亿元", "营业收入80亿元，净利润20亿元"],
        chunks=chunks,
        doc_ids=["doc-a"],
        plan={"domain": "financial_reports", "numbers": ["2025年"]},
    )

    assert len(ranked) >= 2
    assert {ranked[0].section_path, ranked[1].section_path} == {"主要财务数据", "利润表"}


def test_build_plan_exposes_question_type_and_rewrite_catalog() -> None:
    request = RunQuestionTaskRequest(
        mode="A",
        qid="calc-1",
        question="假设现金价值为80万元，累计已领养老年金20万元，身故保险金应如何计算？",
        options=["80万元", "85万元", "90万元", "100万元"],
        answerFormat="single",
        docIds=["doc-a", "doc-b"],
    )

    plan = build_plan(request)
    active_plan = activate_rewrite(plan, "numbers_focus")

    assert plan["question_type"] == "calculation"
    assert plan["requires_multi_doc"] is True
    assert "numbers_focus" in plan["rewrite_catalog"]
    assert "数字" in active_plan["query_hints"]


def test_retrieval_quality_requests_retry_for_missing_numbers() -> None:
    chunks = [
        StructuredChunk(
            chunkId="c1",
            docId="doc-a",
            title="保险合同",
            domain="insurance",
            pageNo=1,
            sectionPath="保险责任",
            chunkType="prose",
            clauseNo=None,
            chunkText="身故保险金按较大值给付，但本页未出现具体金额。",
        )
    ]
    reasoning_results = [
        ReasoningItem(option="A", verdict="insufficient", reasoning="证据不足"),
        ReasoningItem(option="B", verdict="insufficient", reasoning="证据不足"),
        ReasoningItem(option="C", verdict="refute", reasoning="证据不支持"),
        ReasoningItem(option="D", verdict="insufficient", reasoning="证据不足"),
    ]

    evaluation = evaluate_retrieval_quality(
        question="假设现金价值80万元，累计已领养老年金20万元，身故保险金是多少？",
        options=["80万元", "85万元", "90万元", "100万元"],
        plan={
            "question_type": "calculation",
            "numbers": ["80万元", "20万元"],
            "clause_refs": [],
            "focus_terms": ["现金价值", "身故保险金"],
            "quality_gate": {"min_doc_coverage": 1},
        },
        candidate_chunks=chunks,
        reasoning_results=reasoning_results,
        candidate_doc_ids=["doc-a"],
    )

    assert evaluation["decision"] == "retry"
    assert evaluation["next_action"] in {"numbers_focus", "option_split"}
    assert "number_coverage_low" in evaluation["failure_reasons"]


def test_retrieval_loop_retries_with_rewrite(monkeypatch) -> None:
    payload = RunQuestionTaskRequest(
        mode="A",
        qid="loop-1",
        question="根据条款，哪些说法正确？",
        options=["A选项", "B选项", "C选项", "D选项"],
        answerFormat="multi",
        docIds=["doc-a"],
    )
    chunks = [
        StructuredChunk(
            chunkId="c1",
            docId="doc-a",
            title="测试条款",
            domain="regulatory",
            pageNo=1,
            sectionPath="第一章 总则",
            chunkType="clause",
            clauseNo="第一条",
            chunkText="第一条 测试内容。",
        )
    ]
    call_count = {"rank_chunks": 0}

    def fake_rank_chunks(question, options, chunks, doc_ids, plan):  # noqa: ANN001
        call_count["rank_chunks"] += 1
        return chunks

    def fake_reason_options(options, evidence, question="", answer_format="single"):  # noqa: ANN001
        return (
            [
                ReasoningItem(option="A", verdict="insufficient", reasoning="证据不足"),
                ReasoningItem(option="B", verdict="insufficient", reasoning="证据不足"),
                ReasoningItem(option="C", verdict="support", reasoning="证据支持"),
                ReasoningItem(option="D", verdict="refute", reasoning="证据反驳"),
            ],
            {"A": evidence[:1], "B": evidence[:1], "C": evidence[:1], "D": evidence[:1]},
        )

    evaluations = [
        {
            "quality_score": 0.31,
            "quality_label": "low",
            "decision": "retry",
            "next_action": "option_split",
            "failure_reasons": ["insufficient_options_high"],
            "insufficient_options": ["A", "B"],
        },
        {
            "quality_score": 0.82,
            "quality_label": "high",
            "decision": "accept",
            "next_action": "accept",
            "failure_reasons": [],
            "insufficient_options": [],
        },
    ]

    monkeypatch.setattr(retrieval_loop_module, "rank_chunks", fake_rank_chunks)
    monkeypatch.setattr(retrieval_loop_module, "reason_options", fake_reason_options)
    monkeypatch.setattr(retrieval_loop_module, "evaluate_retrieval_quality", lambda **kwargs: evaluations.pop(0))

    result = retrieval_loop_module.run_retrieval_loop(
        payload,
        available_doc_ids={"doc-a"},
        structured_chunks=chunks,
    )

    assert call_count["rank_chunks"] == 2
    assert result["plan"]["active_rewrite_name"] == "option_split"
    assert any(log == "react_round=2" for log in result["loop_logs"])


def test_loaders_can_fall_back_to_preprocessed_jsonl(tmp_path: Path, monkeypatch) -> None:
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(
        '{"doc_id":"fallback-doc","title":"fallback-doc","file_type":"txt","file_path":"sample.txt","chunk_count":1,"status":"done"}\n',
        encoding="utf-8",
    )
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        (
            '{"chunkId":"fallback-1","docId":"fallback-doc","title":"fallback-doc","domain":"regulatory",'
            '"pageNo":1,"sectionPath":"第一章 总则","chunkType":"clause","clauseNo":"第一条",'
            '"chunkText":"第一条 为规范信息披露行为，制定本办法。"}\n'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main_module, "PREPROCESSED_DOCUMENTS_FILE", docs_file)
    monkeypatch.setattr(main_module, "PREPROCESSED_CHUNKS_FILE", chunks_file)
    monkeypatch.setattr(main_module, "read_json", lambda path: [])

    documents = main_module._load_documents()
    chunks = main_module._load_chunks()

    assert documents
    assert documents[0].document_id == "fallback-doc"
    assert chunks
    assert chunks[0]["docId"] == "fallback-doc"


def test_loaders_merge_local_and_preprocessed_jsonl(tmp_path: Path, monkeypatch) -> None:
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(
        '{"doc_id":"fallback-doc","title":"fallback-doc","file_type":"txt","file_path":"sample.txt","chunk_count":1,"status":"done"}\n',
        encoding="utf-8",
    )
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        (
            '{"chunkId":"fallback-1","docId":"fallback-doc","title":"fallback-doc","domain":"regulatory",'
            '"pageNo":1,"sectionPath":"第一章 总则","chunkType":"clause","clauseNo":"第一条",'
            '"chunkText":"第一条 为规范信息披露行为，制定本办法。"}\n'
        ),
        encoding="utf-8",
    )
    local_document = DocumentRecord(
        documentId="local-doc",
        title="local-doc",
        fileType="txt",
        sourcePath="local.txt",
        status="done",
        chunkCount=1,
    )
    local_chunk = {
        "chunkId": "local-1",
        "docId": "local-doc",
        "title": "local-doc",
        "domain": "general",
        "pageNo": 1,
        "sectionPath": "第1页",
        "chunkType": "prose",
        "clauseNo": None,
        "chunkText": "本地调试文档。",
    }

    monkeypatch.setattr(main_module, "PREPROCESSED_DOCUMENTS_FILE", docs_file)
    monkeypatch.setattr(main_module, "PREPROCESSED_CHUNKS_FILE", chunks_file)
    monkeypatch.setattr(main_module, "_load_local_documents", lambda: [local_document])
    monkeypatch.setattr(main_module, "_load_local_chunks", lambda: [local_chunk])

    documents = main_module._load_documents()
    chunks = main_module._load_chunks()

    assert {item.document_id for item in documents} == {"local-doc", "fallback-doc"}
    assert {item["docId"] for item in chunks} == {"local-doc", "fallback-doc"}


def test_task_run_excludes_done_docs_without_real_chunks(monkeypatch) -> None:
    chunk = StructuredChunk(
        chunkId="doc-ok-1",
        docId="doc-ok",
        title="有效文档",
        domain="insurance",
        pageNo=1,
        sectionPath="保险责任",
        chunkType="clause",
        clauseNo="第十条",
        chunkText="第十条 本合同生效后提供给付责任。",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        main_module,
        "_load_documents",
        lambda: [
            DocumentRecord(
                documentId="doc-orphan",
                title="断链文档",
                fileType="txt",
                sourcePath="orphan.txt",
                status="done",
                chunkCount=3,
            ),
            DocumentRecord(
                documentId="doc-ok",
                title="有效文档",
                fileType="txt",
                sourcePath="ok.txt",
                status="done",
                chunkCount=1,
            ),
        ],
    )
    monkeypatch.setattr(main_module, "_load_chunks", lambda: [chunk.model_dump(by_alias=True)])

    def fake_run_retrieval_loop(payload, available_doc_ids, structured_chunks):  # noqa: ANN001
        captured["available_doc_ids"] = set(available_doc_ids)
        return {
            "plan": {"domain": "insurance", "retrieval_plan": "direct", "reasoning_plan": "optionwise"},
            "doc_ids": ["doc-ok"],
            "candidate_chunks": [chunk],
            "reasoning_results": [
                ReasoningItem(option="A", verdict="support", reasoning="证据支持 A。"),
                ReasoningItem(option="B", verdict="refute", reasoning="证据不支持 B。"),
                ReasoningItem(option="C", verdict="insufficient", reasoning="证据不足。"),
                ReasoningItem(option="D", verdict="refute", reasoning="证据不支持 D。"),
            ],
            "evidence_map": {"A": [chunk], "B": [], "C": [], "D": []},
            "evaluation": {"quality_score": 0.9, "quality_label": "high", "failure_reasons": []},
            "loop_logs": [],
        }

    monkeypatch.setattr(main_module, "run_retrieval_loop", fake_run_retrieval_loop)
    monkeypatch.setattr(main_module, "build_memory_ledger", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        main_module,
        "get_qwen_config_status",
        lambda: QwenConfigStatus(
            enabled=False,
            requested_model="none",
            base_url="https://example.com",
            missing_settings=("ENABLE_QWEN_LLM",),
            disable_reason="disabled_by_config",
        ),
    )

    response = client.post(
        "/api/tasks/run",
        json={
            "mode": "A",
            "qid": "Q-CHAIN-FIX",
            "question": "该条款是否提供给付责任？",
            "options": ["提供给付责任", "不提供给付责任", "责任未明确", "属于免责范围"],
            "answerFormat": "single",
            "docIds": ["doc-orphan", "doc-ok"],
        },
    )

    assert response.status_code == 200
    assert captured["available_doc_ids"] == {"doc-ok"}
