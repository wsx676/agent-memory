"""Per-option verification reasoner with multi-signal scoring.

Replaces the original simple overlap-based reasoner with:
- Multi-signal evidence scoring (keyword + clause + numeric + section)
- Improved negation detection (regex-based, not substring)
- Numeric extraction and matching for calculation questions
- Confidence scoring (0.0-1.0) per option
- Question-aware evidence binding
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from api.models import ReasoningItem, StructuredChunk
from api.services.retriever import score_overlap, tokenize

OPTION_NAMES = ("A", "B", "C", "D", "E", "F")

# Negation patterns (regex, not substring)
NEGATION_PATTERNS = [
    re.compile(r"不得(?!.*但)"),
    re.compile(r"不予"),
    re.compile(r"不包括"),
    re.compile(r"不包含"),
    re.compile(r"不属于"),
    re.compile(r"不应"),
    re.compile(r"不能"),
    re.compile(r"不可"),
    re.compile(r"除了?"),
    re.compile(r"除外"),
    re.compile(r"不视为"),
    re.compile(r"免于"),
    re.compile(r"免除"),
    re.compile(r"无须"),
    re.compile(r"不适用"),
]

# Exception/override patterns that may reverse a negation
EXCEPTION_PATTERNS = [
    re.compile(r"但.*(?:应当|应|可以|允许)"),
    re.compile(r"另有规定.*(?:从|按)"),
    re.compile(r"除.*外.*(?:属于|包括|适用)"),
]

# Numeric extraction patterns
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
AMOUNT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:万元|亿元|元|万|亿)")
RATIO_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:倍|‰|万分之)")

# Calculation question indicators
CALC_KEYWORDS = re.compile(r"(?:计算|比例|增长率|增幅|占比|合计|总计|差额|相差|倍数|排序|排名|从高到低|从低到高|由大到小|由小到大)")


@dataclass(slots=True)
class OptionAnalysis:
    option: str
    option_name: str
    verdict: str
    confidence: float
    reasoning: str
    top_chunks: list[StructuredChunk]
    matched_numbers: list[str]
    has_negation: bool


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric values from text."""
    return NUMBER_PATTERN.findall(text)


# 选项核心内容标记的正则：数值/百分号/金额/比率——这些是证据必须命中的硬标记
_OPTION_NUMERIC_MARKERS = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|万元|亿元|元|万|亿|倍|‰|分之)|\d+(?:\.\d+)?"
)


def _extract_option_markers(option: str) -> set[str]:
    """提取选项中的"核心内容标记"——数值、百分号、金额等。

    这些标记是判断证据是否实质匹配的关键：如果选项含具体数值，
    但 top 证据中一个都没命中，说明证据和选项缺乏实质对应，
    应判 insufficient 而非凭宽泛的词法重合判 support/refute。
    """
    return set(_OPTION_NUMERIC_MARKERS.findall(option))


def _check_evidence_substance(
    option: str,
    top_evidence_text: str,
) -> tuple[bool, float]:
    """检查 top 证据与选项的实质匹配度。

    返回 (has_substance, hit_ratio):
    - has_substance: 证据是否实质命中了选项的核心内容
    - hit_ratio: 选项标记在证据中的命中率（0~1）
    """
    markers = _extract_option_markers(option)
    if not markers:
        # 选项不含数值标记——用关键实体词判断
        option_tokens = set(t for t in tokenize(option) if len(t) >= 2)
        if not option_tokens:
            return True, 1.0  # 无法提取标记，不阻拦
        hit = sum(1 for t in option_tokens if t in top_evidence_text)
        ratio = hit / len(option_tokens)
        return ratio >= 0.3, ratio

    # 选项含数值标记——至少命中 1 个才算实质匹配
    hit = 0
    for marker in markers:
        # 数值标记做宽松匹配（去掉空格后子串包含）
        compact_marker = re.sub(r"\s+", "", marker)
        compact_text = re.sub(r"\s+", "", top_evidence_text)
        if compact_marker in compact_text:
            hit += 1
    ratio = hit / len(markers) if markers else 1.0
    return hit > 0, ratio


def _check_negation(text: str) -> tuple[bool, bool]:
    """Check if text contains negation, and if there's an exception override.

    Returns (has_negation, has_exception).
    """
    has_neg = any(pattern.search(text) for pattern in NEGATION_PATTERNS)
    has_exc = any(pattern.search(text) for pattern in EXCEPTION_PATTERNS)
    return has_neg, has_exc


def _score_evidence_for_option(
    option: str,
    question: str,
    evidence: list[StructuredChunk],
) -> list[tuple[float, StructuredChunk]]:
    """Score each evidence chunk for relevance to a specific option.
    
    Multi-signal scoring:
    - Keyword overlap (primary)
    - Question term boost (if evidence also matches question keywords)
    - Clause number matching
    - Numeric value matching
    - Section path matching
    """
    option_tokens = set(tokenize(option))
    question_tokens = set(tokenize(question))
    option_numbers = set(_extract_numbers(option))
    
    scored: list[tuple[float, StructuredChunk]] = []
    for chunk in evidence:
        text = chunk.chunk_text
        # Base: keyword overlap between option and chunk
        base_score = score_overlap(option, text)
        
        # Boost: if chunk also matches question keywords
        question_overlap = score_overlap(question, text)
        question_boost = question_overlap * 0.3
        
        # Clause number matching
        clause_boost = 0.0
        if chunk.clause_no and option_numbers:
            clause_numbers = set(_extract_numbers(chunk.clause_no or ""))
            if clause_numbers & option_numbers:
                clause_boost = 0.2
        
        # Numeric value matching
        numeric_boost = 0.0
        if option_numbers:
            chunk_numbers = set(_extract_numbers(text))
            matched = option_numbers & chunk_numbers
            if matched:
                numeric_boost = min(0.15 * len(matched), 0.3)
        
        # Section path matching
        section_boost = 0.0
        if chunk.section_path and option_tokens:
            section_tokens = set(tokenize(chunk.section_path))
            common = option_tokens & section_tokens
            if common:
                section_boost = min(0.05 * len(common), 0.15)
        
        total = base_score + question_boost + clause_boost + numeric_boost + section_boost
        scored.append((total, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _determine_verdict(
    option: str,
    question: str,
    scored_evidence: list[tuple[float, StructuredChunk]],
) -> tuple[str, float, list[str]]:
    """Determine verdict for an option based on scored evidence.

    三层 insufficient 检查（激活多轮补检索的前提）：
    1. 绝对阈值：avg_score < 0.2 → insufficient
    2. 实质匹配：选项含数值标记但 top 证据一个都没命中 → insufficient
    3. 相对阈值：由 reason_options 做横向比较后回填（见 _apply_relative_threshold）

    Returns (verdict, confidence, matched_numbers).
    """
    if not scored_evidence or scored_evidence[0][0] < 0.1:
        return "insufficient", 0.1, []

    # Take top-3 evidence
    top = scored_evidence[:3]
    avg_score = sum(s for s, _ in top) / len(top)
    top_text = " ".join(chunk.chunk_text for _, chunk in top)

    # Extract matched numbers
    option_numbers = set(_extract_numbers(option))
    chunk_numbers = set(_extract_numbers(top_text))
    matched_numbers = sorted(option_numbers & chunk_numbers)

    # ── insufficient 检查 1：绝对阈值（从 0.15 提高到 0.2）──
    if avg_score < 0.2:
        return "insufficient", 0.2, matched_numbers

    # ── insufficient 检查 2：实质匹配度 ──
    # 选项含数值/百分号等硬标记，但 top 证据一个都没命中 → 证据和选项缺乏实质对应
    has_substance, hit_ratio = _check_evidence_substance(option, top_text)
    if not has_substance:
        return "insufficient", 0.25, matched_numbers

    # Check negation in top evidence
    has_neg, has_exc = _check_negation(top_text)

    # Determine verdict
    if has_neg and not has_exc:
        # Evidence contains negation that may contradict the option
        option_tokens = set(tokenize(option))
        neg_context = _find_negation_context(top_text, option_tokens)
        if neg_context:
            verdict = "refute"
            confidence = min(0.5 + avg_score * 0.5, 0.85)
        else:
            verdict = "support"
            confidence = min(0.4 + avg_score * 0.4, 0.7)
    else:
        verdict = "support"
        confidence = min(0.4 + avg_score * 0.6, 0.9)

    # Boost confidence if numbers match
    if matched_numbers and verdict == "support":
        confidence = min(confidence + 0.1, 0.95)

    return verdict, round(confidence, 2), matched_numbers


def _find_negation_context(text: str, option_tokens: set[str]) -> bool:
    """Check if negation in text is near option tokens (within 50 chars)."""
    for pattern in NEGATION_PATTERNS:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            context_tokens = set(tokenize(context))
            if option_tokens & context_tokens:
                return True
    return False


def _generate_reasoning(
    option_name: str,
    option: str,
    verdict: str,
    confidence: float,
    top_chunks: list[StructuredChunk],
    matched_numbers: list[str],
    has_negation: bool,
) -> str:
    """Generate human-readable reasoning text."""
    parts = [f"选项{option_name}判定：{verdict}（置信度{confidence:.0%}）"]
    
    if top_chunks:
        best_chunk = top_chunks[0]
        loc = f"doc={best_chunk.doc_id}, p{best_chunk.page_no}"
        if best_chunk.clause_no:
            loc += f", 条款{best_chunk.clause_no}"
        parts.append(f"关键证据[{loc}]")
        snippet = re.sub(r"\s+", " ", best_chunk.chunk_text).strip()[:150]
        parts.append(f"证据摘要：{snippet}")
    
    if matched_numbers:
        parts.append(f"数值匹配：{','.join(matched_numbers)}")
    
    if has_negation and verdict == "refute":
        parts.append("检测到否定表述，判定为反驳")
    elif has_negation and verdict == "support":
        parts.append("虽含否定词但非针对本选项")
    
    if verdict == "insufficient":
        parts.append("证据不足，无法确认")
    
    return "；".join(parts)


def reason_options(
    options: list[str],
    evidence: list[StructuredChunk],
    question: str = "",
    answer_format: str = "single",
) -> tuple[list[ReasoningItem], dict[str, list[StructuredChunk]]]:
    """Run per-option verification on all options.

    三阶段判定：
    1. 逐选项独立判定（_determine_verdict：绝对阈值 + 实质匹配）
    2. 横向相对置信度比较（远低于中位数的选项降级为 insufficient）
    3. 计算题特殊处理（数值匹配可提升 insufficient→support）

    Args:
        options: List of option texts (A, B, C, D, ...)
        evidence: List of candidate evidence chunks
        question: The question text (for question-aware matching)
        answer_format: "single", "multi", or "judge"

    Returns:
        Tuple of (reasoning items, evidence map by option name)
    """
    import statistics as _stats

    results: list[ReasoningItem] = []
    evidence_map: dict[str, list[StructuredChunk]] = {}

    is_calc_question = bool(CALC_KEYWORDS.search(question))

    # ── 阶段 1+2：逐选项打分并收集 top-1 分数用于横向比较 ──
    pending: list[dict[str, object]] = []
    for option_name, option in zip(OPTION_NAMES, options):
        scored = _score_evidence_for_option(option, question, evidence)
        top_chunks = [chunk for _, chunk in scored[:3]]
        verdict, confidence, matched_numbers = _determine_verdict(option, question, scored)
        top_text = " ".join(chunk.chunk_text for chunk in top_chunks)
        has_neg, _ = _check_negation(top_text)
        top1_score = scored[0][0] if scored else 0.0
        pending.append({
            "option_name": option_name,
            "option": option,
            "verdict": verdict,
            "confidence": confidence,
            "matched_numbers": matched_numbers,
            "top_chunks": top_chunks,
            "has_neg": has_neg,
            "top1_score": top1_score,
        })
        evidence_map[option_name] = top_chunks

    # ── 阶段 2：横向相对置信度比较 ──
    # 如果某选项的 top1_score 远低于同题中位数（< 35%），说明证据明显不匹配，
    # 即使绝对分超过阈值也降级为 insufficient——这能激活多轮补检索。
    all_scores = [p["top1_score"] for p in pending if p["top1_score"] > 0]
    if len(all_scores) >= 3:
        score_median = _stats.median(all_scores)
        if score_median > 0.1:
            for p in pending:
                if p["verdict"] != "insufficient" and p["top1_score"] < score_median * 0.35:
                    p["verdict"] = "insufficient"
                    p["confidence"] = 0.25

    # ── 阶段 3：计算题特殊处理 + 生成推理文本 ──
    for p in pending:
        verdict = p["verdict"]
        confidence = p["confidence"]
        matched_numbers = p["matched_numbers"]

        # For calculation questions, boost insufficient to support if numbers match
        if is_calc_question and verdict == "insufficient" and matched_numbers:
            verdict = "support"
            confidence = max(confidence, 0.5)

        reasoning = _generate_reasoning(
            p["option_name"], p["option"], verdict, confidence,
            p["top_chunks"], matched_numbers, p["has_neg"],
        )

        results.append(
            ReasoningItem(
                option=p["option_name"],
                verdict=verdict,  # type: ignore[arg-type]
                reasoning=reasoning,
            )
        )

    return results, evidence_map
