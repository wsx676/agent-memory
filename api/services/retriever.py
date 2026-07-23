from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from api.models import StructuredChunk


FIELD_WEIGHTS = {
    "title": 4.0,
    "section_path": 3.0,
    "clause_no": 5.0,
    "chunk_text": 1.0,
}
TOP_K_DOCS = 8
TOP_K_CHUNKS = 12
SECTION_HINTS = (
    "风险提示",
    "盈利预测",
    "投资建议",
    "主要财务数据",
    "资产负债表",
    "利润表",
    "现金流量表",
    "保险责任",
    "免责",
    "给付",
    "现金价值",
    "受益所有人",
    "客户尽职调查",
    "信息披露",
    "募集说明书",
    "发行条款",
    "偿付顺序",
    "评级",
)
DOMAIN_ALIASES = {
    "report": "financial_reports",
    "financial_report": "financial_reports",
    "contract": "financial_contracts",
    "financial_contract": "financial_contracts",
}
STOPWORDS = {
    "以下",
    "哪些",
    "下列",
    "相关",
    "规定",
    "根据",
    "关于",
    "结合",
    "是否",
    "其中",
    "说法",
    "描述",
    "正确",
    "错误",
    "符合",
    "内容",
    "要求",
    "情况",
    "以及",
    "或者",
    "并且",
    "可以",
    "应当",
    "不得",
    "进行",
    "公司",
    "机构",
}
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?|\d+年|\d+月|\d+日")
CLAUSE_PATTERN = re.compile(r"第[\d一二三四五六七八九十百千万]+条")
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9.%]+")
# 硬编码同义词组作为回退（当data/synonym_dict.json加载失败时使用）
QUERY_SYNONYM_GROUPS_FALLBACK = (
    ("营业收入", "营收"),
    ("归母净利润", "归属于上市公司股东的净利润"),
    ("责任免除", "免责"),
    ("现金流量表", "现金流"),
    ("客户尽职调查", "尽职调查"),
    ("受益所有人", "最终受益人"),
    ("退保费用", "退保手续费"),
)


@lru_cache(maxsize=1)
def _load_synonym_dict() -> dict[str, list[list[str]]]:
    """加载领域同义词词典 data/synonym_dict.json，按domain+全局_global合并同义词组。"""
    dict_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synonym_dict.json")
    try:
        with open(dict_path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k != "_meta"}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_synonym_groups(domain: str) -> list[tuple[str, ...]]:
    """获取domain对应的同义词组（合并_global + 领域特定），词典缺失时回退到硬编码组。"""
    synonym_dict = _load_synonym_dict()
    if not synonym_dict:
        return list(QUERY_SYNONYM_GROUPS_FALLBACK)
    groups: list[tuple[str, ...]] = []
    groups.extend(tuple(g) for g in synonym_dict.get("_global", []))
    if domain and domain != "general":
        groups.extend(tuple(g) for g in synonym_dict.get(domain, []))
    return groups


@dataclass(frozen=True)
class QueryContext:
    domain: str
    query_text: str
    question_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    option_tokens: tuple[str, ...]
    numbers: tuple[str, ...]
    clause_refs: tuple[str, ...]
    section_terms: tuple[str, ...]
    exact_phrases: tuple[str, ...]
    high_value_terms: tuple[str, ...]


@dataclass(frozen=True)
class CorpusStatistics:
    doc_count: int
    avg_lengths: dict[str, float]
    doc_freqs: dict[str, dict[str, int]]


def _normalize_domain(domain: str | None) -> str:
    if not domain:
        return "general"
    lowered = domain.lower()
    return DOMAIN_ALIASES.get(lowered, lowered)


@lru_cache(maxsize=200_000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for matched in TOKEN_PATTERN.finditer(text.lower()):
        token = matched.group(0)
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            compact = token.strip()
            if len(compact) <= 8:
                tokens.append(compact)
            if len(compact) == 1:
                tokens.append(compact)
                continue
            for index in range(len(compact) - 1):
                tokens.append(compact[index : index + 2])
            if len(compact) <= 4:
                for index in range(len(compact) - 2):
                    tokens.append(compact[index : index + 3])
        else:
            tokens.append(token)
    return tuple(token for token in tokens if len(token) > 1 and token not in STOPWORDS)


def tokenize(text: str) -> list[str]:
    return list(_tokenize_cached(text))


def _extract_numbers(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(NUMBER_PATTERN.findall(text)))


def _extract_clause_refs(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(CLAUSE_PATTERN.findall(text)))


def _extract_section_terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in SECTION_HINTS if term in text)


def _normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _expand_query_tokens(query_text: str, base_tokens: tuple[str, ...], domain: str = "general") -> tuple[str, ...]:
    expanded = list(base_tokens)
    lowered = query_text.lower()
    for group in _get_synonym_groups(domain):
        if any(term.lower() in lowered for term in group):
            for term in group:
                expanded.extend(tokenize(term))
    return tuple(dict.fromkeys(expanded))


def _extract_exact_phrases(text: str) -> tuple[str, ...]:
    phrases: list[str] = []
    for matched in re.findall(r"[\u4e00-\u9fff]{4,}|[A-Za-z0-9.%]{3,}", text):
        cleaned = matched.strip()
        if cleaned and cleaned not in STOPWORDS:
            phrases.append(cleaned)
    return tuple(dict.fromkeys(phrases))[:16]


def _extract_high_value_terms(
    query_text: str,
    query_tokens: tuple[str, ...],
    numbers: tuple[str, ...],
    clause_refs: tuple[str, ...],
    section_terms: tuple[str, ...],
) -> tuple[str, ...]:
    values = [*clause_refs, *numbers, *section_terms, *query_tokens, query_text]
    deduped = [value for value in dict.fromkeys(values) if value and len(value) > 1 and value not in STOPWORDS]
    return tuple(deduped[:24])


def _build_query_context(
    question: str,
    options: list[str] | None = None,
    plan: dict[str, object] | None = None,
    candidate_chunks: list[StructuredChunk] | None = None,
) -> QueryContext:
    options = options or []
    query_rewrite = str((plan or {}).get("query_rewrite", "") or "").strip()
    query_hints = [str(item) for item in ((plan or {}).get("query_hints") or []) if str(item).strip()]
    focused_options = [str(item) for item in ((plan or {}).get("option_focus") or options) if str(item).strip()]
    focus_terms = tuple(str(item) for item in ((plan or {}).get("focus_terms") or []) if str(item).strip())
    query_text = " ".join([question, query_rewrite, *query_hints, *focused_options]).strip()
    question_tokens = tuple(tokenize(question))
    base_query_tokens = tuple(tokenize(query_text))
    domain = _normalize_domain(str((plan or {}).get("domain", "") or ""))
    if domain == "general" and candidate_chunks:
        inferred = Counter(_normalize_domain(chunk.domain) for chunk in candidate_chunks if chunk.domain)
        if inferred:
            domain = inferred.most_common(1)[0][0]
    query_tokens = _expand_query_tokens(query_text, base_query_tokens, domain)
    option_tokens = tuple(token for option in focused_options for token in tokenize(option))
    numbers = tuple((plan or {}).get("numbers", ()) or _extract_numbers(query_text))
    clause_refs = tuple((plan or {}).get("clause_refs", ()) or _extract_clause_refs(query_text))
    section_terms = tuple(dict.fromkeys([*_extract_section_terms(query_text), *focus_terms]))
    exact_phrases = _extract_exact_phrases(query_text)
    high_value_terms = _extract_high_value_terms(query_text, query_tokens, numbers, clause_refs, section_terms)
    return QueryContext(
        domain=domain,
        query_text=query_text,
        question_tokens=question_tokens,
        query_tokens=query_tokens,
        option_tokens=option_tokens,
        numbers=numbers,
        clause_refs=clause_refs,
        section_terms=section_terms,
        exact_phrases=exact_phrases,
        high_value_terms=high_value_terms,
    )


def _chunk_field_text(chunk: StructuredChunk, field_name: str) -> str:
    if field_name == "title":
        return chunk.title
    if field_name == "section_path":
        return chunk.section_path
    if field_name == "clause_no":
        return chunk.clause_no or ""
    return chunk.chunk_text


def _build_statistics(chunks: list[StructuredChunk]) -> CorpusStatistics:
    doc_freqs: dict[str, dict[str, int]] = {field: defaultdict(int) for field in FIELD_WEIGHTS}
    total_lengths: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        for field_name in FIELD_WEIGHTS:
            tokens = tokenize(_chunk_field_text(chunk, field_name))
            total_lengths[field_name] += len(tokens)
            for token in set(tokens):
                doc_freqs[field_name][token] += 1
    doc_count = max(len(chunks), 1)
    avg_lengths = {
        field_name: (total_lengths[field_name] / doc_count) if total_lengths[field_name] else 1.0
        for field_name in FIELD_WEIGHTS
    }
    return CorpusStatistics(
        doc_count=doc_count,
        avg_lengths=avg_lengths,
        doc_freqs={field: dict(freqs) for field, freqs in doc_freqs.items()},
    )


def _bm25_score_tokens(
    query_tokens: tuple[str, ...],
    text_tokens: list[str],
    *,
    doc_count: int,
    avg_length: float,
    doc_freqs: dict[str, int],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    term_freqs = Counter(text_tokens)
    query_counts = Counter(query_tokens)
    doc_length = len(text_tokens)
    avg_length = max(avg_length, 1.0)
    score = 0.0
    for token, query_weight in query_counts.items():
        frequency = term_freqs.get(token, 0)
        if not frequency:
            continue
        document_frequency = doc_freqs.get(token, 0)
        idf = math.log(1.0 + (doc_count - document_frequency + 0.5) / (document_frequency + 0.5))
        denominator = frequency + k1 * (1.0 - b + b * (doc_length / avg_length))
        score += idf * ((frequency * (k1 + 1.0)) / denominator) * (1.0 + 0.15 * (query_weight - 1.0))
    return score


def _base_chunk_score(chunk: StructuredChunk, context: QueryContext, stats: CorpusStatistics) -> float:
    score = 0.0
    for field_name, weight in FIELD_WEIGHTS.items():
        field_tokens = tokenize(_chunk_field_text(chunk, field_name))
        field_score = _bm25_score_tokens(
            context.query_tokens,
            field_tokens,
            doc_count=stats.doc_count,
            avg_length=stats.avg_lengths[field_name],
            doc_freqs=stats.doc_freqs[field_name],
        )
        score += field_score * weight
    return score


def _count_text_hits(text: str, values: tuple[str, ...]) -> int:
    return sum(1 for value in values if value and value in text)


def _matched_query_signals(chunk: StructuredChunk, context: QueryContext) -> set[str]:
    text = _normalize_search_text(
        " ".join((chunk.title or "", chunk.section_path or "", chunk.clause_no or "", chunk.chunk_text or ""))
    )
    matched: set[str] = set()
    for value in (*context.clause_refs, *context.numbers, *context.section_terms, *context.exact_phrases, *context.high_value_terms):
        normalized = _normalize_search_text(value)
        if normalized and normalized in text:
            matched.add(value)
    return matched


def _rule_boost(chunk: StructuredChunk, context: QueryContext) -> float:
    score = 0.0
    section_path = chunk.section_path or ""
    chunk_text = chunk.chunk_text or ""
    title = chunk.title or ""
    normalized_text = _normalize_search_text(" ".join((title, section_path, chunk.clause_no or "", chunk_text)))

    phrase_hits = sum(
        1 for phrase in context.exact_phrases if len(phrase) >= 4 and _normalize_search_text(phrase) in normalized_text
    )
    if phrase_hits:
        score += min(phrase_hits, 2) * 2.5

    matched_terms = sum(
        1 for term in context.high_value_terms if _normalize_search_text(term) and _normalize_search_text(term) in normalized_text
    )
    if matched_terms:
        score += min(matched_terms, 6) * 0.7
        if matched_terms >= 2:
            score += 1.0

    if context.clause_refs:
        if chunk.clause_no and chunk.clause_no in context.clause_refs:
            score += 8.0
        else:
            score += 3.0 * _count_text_hits(section_path, context.clause_refs)
            score += 2.0 * _count_text_hits(chunk_text, context.clause_refs)

    if context.section_terms:
        score += min(_count_text_hits(section_path, context.section_terms), 2) * 2.5
        score += min(_count_text_hits(title, context.section_terms), 1) * 1.5

    if context.numbers:
        matched_numbers = {
            number
            for number in context.numbers
            if number in chunk_text or number in section_path or number in title
        }
        if matched_numbers:
            score += 2.0 + 1.2 * len(matched_numbers)
            if len(matched_numbers) == len(context.numbers):
                score += 2.0

    if context.domain == "regulatory":
        if chunk.chunk_type == "clause":
            score += 2.5
        if any(keyword in section_path for keyword in ("办法", "规定", "条例", "指引", "监管")):
            score += 1.5
    elif context.domain == "financial_reports":
        if chunk.chunk_type == "table":
            score += 3.0
        if any(keyword in section_path for keyword in ("财务", "资产负债", "利润", "现金流")):
            score += 2.0
    elif context.domain == "financial_contracts":
        if chunk.clause_no:
            score += 1.5
        if any(keyword in section_path for keyword in ("发行", "票息", "评级", "偿付", "条款")):
            score += 2.0
    elif context.domain == "insurance":
        if any(keyword in section_path for keyword in ("保险责任", "免责", "给付", "现金价值", "等待期")):
            score += 2.0
    elif context.domain == "research":
        if any(keyword in section_path for keyword in ("风险提示", "盈利预测", "投资建议")):
            score += 2.0

    if chunk.chunk_type == "table" and context.numbers:
        score += 1.2

    if "目录" in section_path or "目录" in chunk_text[:30]:
        score -= 6.0
    if len(chunk_text.strip()) < 20:
        score -= 1.5

    return score


def _candidate_chunks(
    chunks: list[StructuredChunk],
    *,
    doc_ids: list[str] | None,
    domain: str,
) -> list[StructuredChunk]:
    filtered = [chunk for chunk in chunks if not doc_ids or chunk.doc_id in doc_ids]
    normalized_domain = _normalize_domain(domain)
    if normalized_domain == "general":
        return filtered
    domain_filtered = [chunk for chunk in filtered if _normalize_domain(chunk.domain) == normalized_domain]
    return domain_filtered or filtered



def score_overlap(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    query_counts = Counter(query_tokens)
    text_counts = Counter(text_tokens)
    common = sum(min(text_counts[token], count) for token, count in query_counts.items())
    denominator = math.sqrt(sum(value * value for value in query_counts.values())) + 1.0
    return common / denominator


def rank_documents(
    question: str,
    chunks: list[StructuredChunk],
    plan: dict[str, object] | None = None,
) -> list[str]:
    context = _build_query_context(question, plan=plan, candidate_chunks=chunks)
    candidates = _candidate_chunks(chunks, doc_ids=None, domain=context.domain)
    if not candidates:
        return []
    stats = _build_statistics(candidates)
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_signals: dict[str, set[str]] = defaultdict(set)
    grouped_sections: dict[str, set[str]] = defaultdict(set)
    for chunk in candidates:
        chunk_score = _base_chunk_score(chunk, context, stats) + _rule_boost(chunk, context)
        grouped_scores[chunk.doc_id].append(chunk_score)
        grouped_signals[chunk.doc_id].update(_matched_query_signals(chunk, context))
        grouped_sections[chunk.doc_id].add(chunk.section_path or chunk.title or chunk.chunk_id)

    doc_scores: dict[str, float] = {}
    for doc_id, scores in grouped_scores.items():
        top_scores = sorted(scores, reverse=True)[:3]
        doc_scores[doc_id] = (
            top_scores[0]
            + (sum(top_scores[1:]) * 0.35)
            + (0.3 * len(grouped_signals[doc_id]))
            + (0.08 * len(grouped_sections[doc_id]))
        )

    return [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:TOP_K_DOCS]]


def _diversified_top_chunks(scored_chunks: list[tuple[float, StructuredChunk]]) -> list[StructuredChunk]:
    remaining = list(scored_chunks)
    selected: list[StructuredChunk] = []
    doc_counts: Counter[str] = Counter()
    section_counts: Counter[tuple[str, str]] = Counter()
    while remaining and len(selected) < TOP_K_CHUNKS:
        best_index = 0
        best_score: float | None = None
        for index, (score, chunk) in enumerate(remaining):
            section_key = (chunk.doc_id, chunk.section_path or chunk.title or chunk.chunk_id)
            adjusted = score - (0.5 * doc_counts[chunk.doc_id]) - (1.25 * section_counts[section_key])
            if best_score is None or adjusted > best_score:
                best_index = index
                best_score = adjusted
        _, chunk = remaining.pop(best_index)
        selected.append(chunk)
        doc_counts[chunk.doc_id] += 1
        section_counts[(chunk.doc_id, chunk.section_path or chunk.title or chunk.chunk_id)] += 1
    return selected


def rank_chunks(
    question: str,
    options: list[str],
    chunks: list[StructuredChunk],
    doc_ids: list[str] | None = None,
    plan: dict[str, object] | None = None,
) -> list[StructuredChunk]:
    preview_candidates = [chunk for chunk in chunks if not doc_ids or chunk.doc_id in doc_ids]
    context = _build_query_context(question, options, plan=plan, candidate_chunks=preview_candidates or chunks)
    filtered = _candidate_chunks(chunks, doc_ids=doc_ids, domain=context.domain)
    if not filtered:
        return []

    stats = _build_statistics(filtered)
    scored = [
        (_base_chunk_score(chunk, context, stats) + _rule_boost(chunk, context), chunk)
        for chunk in filtered
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return _diversified_top_chunks(scored)
