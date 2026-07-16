from __future__ import annotations

import re

from api.models import RunQuestionTaskRequest


FINANCE_TERMS = {
    "insurance": ["保险", "给付", "免责", "现金价值"],
    "regulatory": ["条例", "决议", "监管", "规定"],
    "financial_reports": ["年报", "营收", "资产", "负债", "现金流", "利润表", "资产负债表"],
    "financial_contracts": ["条款", "票息", "发行", "评级", "偿付顺序", "募集说明书"],
    "research": ["研报", "市场规模", "同比", "预测", "风险提示", "投资建议", "盈利预测"],
}
CALCULATION_HINTS = ("计算", "金额", "合计", "分别是多少", "排序", "高到低", "低到高", "赔付", "退保")
COMPARISON_HINTS = ("对比", "比较", "差异", "哪份", "哪家", "两份文档", "均", "分别", "共同")
CLAUSE_HINTS = ("条款", "条文", "根据", "依据", "规定", "办法", "条例")
SECTION_HINTS = {
    "insurance": ["保险责任", "身故保险金", "责任免除", "现金价值", "等待期", "给付"],
    "regulatory": ["总则", "信息披露", "受益所有人", "客户尽职调查", "适用范围", "监管要求"],
    "financial_reports": ["主要财务数据", "利润表", "资产负债表", "现金流量表", "营业收入", "净利润"],
    "financial_contracts": ["募集说明书", "发行条款", "评级", "受托管理人", "偿付顺序", "发行规模"],
    "research": ["投资建议", "盈利预测", "风险提示", "核心观点", "市场规模"],
}
GENERIC_REWRITE_HINTS = {
    "base": ("保留题干核心术语", ["原文", "关键术语"]),
    "comparison_focus": ("强调跨文档对比与分别抽取", ["分别", "对比", "逐文档核对"]),
    "numbers_focus": ("强调数字、金额、比例和公式依据", ["数字", "金额", "比例", "公式"]),
    "clause_focus": ("优先命中条款号、章节名和原文义务", ["条款原文", "章节标题", "义务", "责任"]),
    "section_focus": ("优先命中高价值章节标题与字段", ["高价值章节", "结构字段", "核心术语"]),
    "option_split": ("按选项逐条验证支持与反驳证据", ["逐选项", "支持证据", "反驳证据"]),
}


def detect_domain(question: str, options: list[str]) -> str:
    merged = f"{question} {' '.join(options)}"
    for domain, keywords in FINANCE_TERMS.items():
        if any(keyword in merged for keyword in keywords):
            return domain
    return "general"


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?%?|\d+年|\d+月|\d+日", text)


def extract_clause_refs(text: str) -> list[str]:
    return re.findall(r"第[\d一二三四五六七八九十百]+条", text)


def detect_question_type(request: RunQuestionTaskRequest) -> str:
    merged = f"{request.question} {' '.join(request.options)}"
    if request.answer_format == "judge":
        return "judge"
    if extract_clause_refs(request.question) or any(hint in merged for hint in CLAUSE_HINTS):
        return "clause_lookup"
    if any(hint in request.question for hint in CALCULATION_HINTS):
        return "calculation"
    if len(request.doc_ids or []) >= 2 or any(hint in merged for hint in COMPARISON_HINTS):
        return "comparison"
    if request.answer_format == "multi":
        return "multi_select"
    return "fact_lookup"


def _extract_focus_terms(question: str, options: list[str], domain: str) -> list[str]:
    merged = f"{question} {' '.join(options)}"
    terms = list(dict.fromkeys(term for term in SECTION_HINTS.get(domain, []) if term in merged))
    if not terms:
        terms = SECTION_HINTS.get(domain, [])[:3]
    return terms


def _build_rewrite_catalog(
    *,
    domain: str,
    question_type: str,
    numbers: list[str],
    clause_refs: list[str],
    focus_terms: list[str],
) -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {}
    for name, (query_rewrite, hints) in GENERIC_REWRITE_HINTS.items():
        catalog[name] = {
            "name": name,
            "query_rewrite": query_rewrite,
            "query_hints": list(hints),
            "focus_terms": list(focus_terms),
        }

    if numbers:
        catalog["numbers_focus"]["query_hints"] = [*catalog["numbers_focus"]["query_hints"], *numbers]
    if clause_refs:
        catalog["clause_focus"]["query_hints"] = [*catalog["clause_focus"]["query_hints"], *clause_refs]
    if focus_terms:
        catalog["section_focus"]["query_hints"] = [*catalog["section_focus"]["query_hints"], *focus_terms]
    if question_type == "calculation":
        catalog["base"]["query_hints"] = [*catalog["base"]["query_hints"], "计算依据", "取较大值", "金额"]
    if question_type == "comparison":
        catalog["comparison_focus"]["query_hints"] = [*catalog["comparison_focus"]["query_hints"], "同一字段对照"]
    if question_type == "clause_lookup":
        catalog["clause_focus"]["query_hints"] = [*catalog["clause_focus"]["query_hints"], "条款定位"]
    if domain == "financial_contracts":
        catalog["comparison_focus"]["query_hints"] = [*catalog["comparison_focus"]["query_hints"], "评级", "发行规模"]
    return catalog


def activate_rewrite(
    plan: dict[str, object],
    rewrite_name: str,
    *,
    option_focus: list[str] | None = None,
) -> dict[str, object]:
    next_plan = dict(plan)
    catalog = dict(plan.get("rewrite_catalog", {}))
    rewrite = dict(catalog.get(rewrite_name, catalog.get("base", {})))
    next_plan["active_rewrite_name"] = rewrite.get("name", rewrite_name)
    next_plan["query_rewrite"] = rewrite.get("query_rewrite", "")
    next_plan["query_hints"] = list(rewrite.get("query_hints", []))
    next_plan["focus_terms"] = list(rewrite.get("focus_terms", plan.get("focus_terms", [])))
    next_plan["option_focus"] = option_focus or []
    return next_plan


def next_rewrite_from_feedback(
    plan: dict[str, object],
    evaluation: dict[str, object],
    options: list[str],
) -> dict[str, object]:
    action = str(evaluation.get("next_action", "accept") or "accept")
    if action == "accept":
        return dict(plan)

    option_names = [str(item) for item in evaluation.get("insufficient_options", []) if str(item)]
    option_focus = [
        option
        for name, option in zip(["A", "B", "C", "D", "E", "F"], options)
        if not option_names or name in option_names
    ]

    rewrite_name = {
        "clause_focus": "clause_focus",
        "numbers_focus": "numbers_focus",
        "comparison_focus": "comparison_focus",
        "option_split": "option_split",
        "section_focus": "section_focus",
        "diversify_sections": "section_focus",
        "broaden_docs": "comparison_focus",
    }.get(action, "base")

    return activate_rewrite(plan, rewrite_name, option_focus=option_focus if action == "option_split" else None)


def build_plan(request: RunQuestionTaskRequest) -> dict[str, object]:
    domain = detect_domain(request.question, request.options)
    numbers = extract_numbers(request.question)
    clause_refs = extract_clause_refs(request.question)
    question_type = detect_question_type(request)
    focus_terms = _extract_focus_terms(request.question, request.options, domain)
    return {
        "domain": domain,
        "answer_format": request.answer_format,
        "known_doc_ids": request.doc_ids or [],
        "numbers": numbers,
        "clause_refs": clause_refs,
        "question_type": question_type,
        "requires_multi_doc": len(request.doc_ids or []) >= 2 or question_type == "comparison",
        "focus_terms": focus_terms,
        "rewrite_catalog": _build_rewrite_catalog(
            domain=domain,
            question_type=question_type,
            numbers=numbers,
            clause_refs=clause_refs,
            focus_terms=focus_terms,
        ),
        "retrieval_plan": "react_lite_chunk_loop" if request.mode == "A" else "react_lite_doc_then_chunk",
        "reasoning_plan": "option_first_with_evidence_gate",
        "max_iterations": 3,
        "quality_gate": {
            "min_chunks": 3,
            "min_doc_coverage": min(max(len(request.doc_ids or []), 1), 2),
        },
    }
