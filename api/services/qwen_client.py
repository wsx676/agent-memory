from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from api.models import ReasoningItem, StructuredChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPTION_NAMES = ("A", "B", "C", "D")


@dataclass(slots=True)
class QwenAnswer:
    answer: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


@dataclass(slots=True)
class QwenConfigStatus:
    enabled: bool
    requested_model: str
    base_url: str
    missing_settings: tuple[str, ...]
    disable_reason: str


def _load_env_file() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_setting(name: str) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    return _load_env_file().get(name)


def _get_bool_setting(name: str, default: bool) -> bool:
    raw = _get_setting(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def qwen_is_configured() -> bool:
    return get_qwen_config_status().enabled


def get_qwen_config_status() -> QwenConfigStatus:
    enabled = str(_get_setting("ENABLE_QWEN_LLM") or "").lower() in {"1", "true", "yes", "on"}
    api_key = _get_setting("DASHSCOPE_API_KEY")
    model_name = _get_setting("MODEL_NAME") or ""
    missing_settings: list[str] = []
    disable_reason = "none"

    if not enabled:
        missing_settings.append("ENABLE_QWEN_LLM")
        disable_reason = "disabled_by_config"
    if not api_key:
        missing_settings.append("DASHSCOPE_API_KEY")
        if disable_reason == "none":
            disable_reason = "missing_api_key"
    if not model_name:
        missing_settings.append("MODEL_NAME")
        if disable_reason == "none":
            disable_reason = "missing_model_name"

    return QwenConfigStatus(
        enabled=enabled and not missing_settings,
        requested_model=model_name or "none",
        base_url=(_get_setting("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        missing_settings=tuple(missing_settings),
        disable_reason=disable_reason,
    )


# Evidence window parameters — balanced per-doc allocation prevents one
# document from monopolising the LLM's context (root cause of 4/11 fc errors).
_EVIDENCE_MIN_PER_DOC = 5           # guarantee at least 5 chunks per doc_id
_EVIDENCE_MAX_TOTAL = 30            # hard cap on total evidence chunks (was 20, raised to avoid dropping key chunks)
_EVIDENCE_SNIPPET_LEN = 1200        # chars per chunk — aligned with merge cap to eliminate truncation
_EVIDENCE_TABLE_SNIPPET_LEN = 1200  # table chunks: same cap but preserve newlines for row structure
_EVIDENCE_MAX_TOTAL_CHARS = 24000   # safety valve: total chars across all chunks (~28.8K tokens/call)

# Priority section keywords — chunks from these sections get boosted in Phase 2
# selection. Identified via truncation analysis: profit distribution plans, revenue
# breakdowns, and cash flow tables are frequently answer-critical but low BM25 score.
_PRIORITY_SECTIONS = (
    "利润分配", "分红", "派发", "现金红利", "每10股", "每股",
    "营业收入", "营业成本", "营收",
    "净利润", "归母",
    "经营活动", "现金流量", "现金流",
    "研发费用", "研发投入",
    "回购", "股份回购",
    "董事会审议", "利润分配预案",
    "资产负债表", "利润表", "现金流量表",
)


def _format_evidence(chunks: list[StructuredChunk]) -> str:
    """Format evidence with balanced per-doc allocation and priority-aware fill.

    Three-phase selection:
    1. Guarantee MIN_PER_DOC chunks per doc (balanced coverage).
    2. Fill remaining slots from leftover — priority sections first, then by
       global BM25 order (chunks list is pre-sorted by rank_chunks).
    3. Safety valve on total chars.

    This prevents answer-critical chunks (e.g. profit distribution plans) from
    being dropped purely because they rank low in BM25 keyword overlap.
    """
    if not chunks:
        return "无候选证据"

    # Group by doc_id, preserving input order within each group
    by_doc: dict[str, list[StructuredChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    # Phase 1: guarantee MIN_PER_DOC slots per doc
    selected: list[StructuredChunk] = []
    seen_ids: set[str] = set()
    for doc_chunks in by_doc.values():
        for chunk in doc_chunks[:_EVIDENCE_MIN_PER_DOC]:
            if chunk.chunk_id not in seen_ids:
                selected.append(chunk)
                seen_ids.add(chunk.chunk_id)

    # Phase 2: fill remaining slots — priority sections first, then global BM25 order.
    # chunks list is pre-sorted by rank_chunks BM25 score, so iterating in order
    # preserves relevance ranking within each group.
    not_selected = [c for c in chunks if c.chunk_id not in seen_ids]
    priority: list[StructuredChunk] = []
    normal: list[StructuredChunk] = []
    for chunk in not_selected:
        section_text = f"{chunk.section_path or ''} {chunk.title or ''} {chunk.chunk_text[:300]}"
        if any(kw in section_text for kw in _PRIORITY_SECTIONS):
            priority.append(chunk)
        else:
            normal.append(chunk)
    # Priority chunks first (within priority, preserve BM25 order), then normal
    for chunk in priority:
        if len(selected) >= _EVIDENCE_MAX_TOTAL:
            break
        selected.append(chunk)
        seen_ids.add(chunk.chunk_id)
    for chunk in normal:
        if len(selected) >= _EVIDENCE_MAX_TOTAL:
            break
        selected.append(chunk)
        seen_ids.add(chunk.chunk_id)

    lines: list[str] = []
    total_chars = 0
    min_required = _EVIDENCE_MIN_PER_DOC * len(by_doc)  # per-doc minimum quota total
    for index, chunk in enumerate(selected, start=1):
        location = f"doc={chunk.doc_id}, page={chunk.page_no}"
        if chunk.clause_no:
            location += f", clause={chunk.clause_no}"
        if chunk.section_path:
            location += f", section={chunk.section_path}"
        # Type-aware truncation: table chunks preserve newlines for row alignment,
        # prose chunks normalise whitespace to save tokens.
        if chunk.chunk_type == "table":
            snippet = chunk.chunk_text.strip()[:_EVIDENCE_TABLE_SNIPPET_LEN]
        else:
            snippet = re.sub(r"\s+", " ", chunk.chunk_text).strip()[:_EVIDENCE_SNIPPET_LEN]
        line = f"{index}. [{location}] {snippet}"
        # Safety valve: stop when total chars budget exceeded AND per-doc minimum met.
        if total_chars + len(line) > _EVIDENCE_MAX_TOTAL_CHARS and len(lines) >= min_required:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines) if lines else "无候选证据"


def _format_reasoning_hints(results: list[ReasoningItem]) -> str:
    """Format per-option analysis with verdict and key evidence summary."""
    lines = []
    for item in results:
        lines.append(f"- 选项{item.option}: {item.verdict}；{item.reasoning[:200]}")
    return "\n".join(lines) if lines else "- 无本地推理提示"


# ── 七维验证框架 + strictness 参数 ───────────────────────────────────
# 来源：优化/reg优化2.md + reg优化3.md
# 通用七维检查清单（所有领域一致）+ 域重点提示（轻量引导）+ 决策严格度（领域相关）

_SEVEN_DIMENSIONS_TEXT = """## 七维对齐验证
对每个选项，依次检查以下七个维度，任一维度存在明确冲突即不选：

1. 实体对齐：证据讨论的主体（公司/人/国家/产品）是否与选项一致？主体不同→冲突。
2. 数值对齐：证据中的数值是否与选项完全一致？正负号、单位、量级均须一致。选项核心数值未在证据出现→证据不足。
3. 方向/趋势对齐：证据描述的变化方向是否与选项一致？"增长"vs"下降/微降/回落"→冲突。"显著增长"vs"先升后降"→冲突。
4. 逻辑结构对齐：证据的逻辑连接方式是否与选项一致？"只有…才"vs"或者"→冲突。"必须"vs"可以"→冲突。排他条件vs并列条件→冲突。
5. 修饰语对齐：选项中的限定词是否在证据中出现？"名义"vs"实际"→冲突。"无条件"vs"有条件"→冲突。选项有限定词但证据未提→证据不足。
6. 来源/角色对齐：证据是权威认定还是当事人申辩？"当事人提出/申辩/认为"→申辩意见，不可单独作为正确依据。"本会认为/经查/决定如下"→监管认定，可作为依据。
7. 范围对齐：证据是否真正回答了选项的问题？文档在参考范围内且证据直接支持→不得以"主题偏移"为由排除。只有选项与题干存在直接逻辑矛盾时才可排除。"""

_DOMAIN_HINTS: dict[str, str] = {
    "research": "本领域常见陷阱：实体混淆(维度1)、数值符号相反(维度2)、方向词矛盾(维度3)、核心数值未出现(维度2)。",
    "regulatory": "本领域常见陷阱：逻辑连接词矛盾(维度4)、申辩冒充认定(维度6)、以主题偏移排除已证选项(维度7)、文末施行日漏读(维度2)。",
    "insurance": "本领域常见陷阱：方向词矛盾(维度3)、逻辑结构矛盾(维度4)。",
    "financial_reports": "本领域常见陷阱：数值单位不一致(维度2)、修饰语缺失(维度5)。",
    "financial_contracts": "本领域常见陷阱：数值不一致(维度2)、逻辑结构矛盾(维度4)。",
}

_DOMAIN_STRICTNESS: dict[str, str] = {
    "research": "high",
    "financial_reports": "high",
    "insurance": "balanced",
    "financial_contracts": "balanced",
    "regulatory": "inclusive",
}


def _build_strictness_rule(strictness: str, answer_format: str) -> str:
    """按 strictness 和 answer_format 构建最终答案规则。"""
    if strictness == "high":
        if answer_format == "multi":
            return ("最终答案规则（严格度: high）：多选题只选七维全部通过且无任何疑似冲突的选项。"
                    "存疑即排除。宁可漏选，不可错选。")
        if answer_format == "tf":
            return ("最终答案规则（严格度: high）：只有七维全部通过且无任何疑似冲突时才选'正确'。"
                    "存疑即选'错误'。")
        return "最终答案规则（严格度: high）：选七维全部通过中证据最强的。存疑时选冲突最少的。"
    if strictness == "inclusive":
        if answer_format == "multi":
            return ("最终答案规则（严格度: inclusive）：多选题只要七维无明确冲突且证据直接支持即纳入。"
                    "不得以'主题偏移/范畴不符'为由排除已证选项。只有某维度存在明确反证时才排除。")
        if answer_format == "tf":
            return ("最终答案规则（严格度: inclusive）：只要七维无明确冲突且证据直接支持即选'正确'。"
                    "只有明确反证才选'错误'。")
        return "最终答案规则（严格度: inclusive）：选证据直接支持的。不得以'主题偏移'为由排除已证选项。"
    # balanced
    if answer_format == "multi":
        return ("最终答案规则（严格度: balanced）：有直接证据支持则选；证据明确排除则不选；"
                "证据部分匹配且无反驳证据的，倾向选择。")
    if answer_format == "tf":
        return "最终答案规则（严格度: balanced）：有直接证据支持即选'正确'；证据明确排除选'错误'；模糊时倾向支持。"
    return "最终答案规则（严格度: balanced）：选证据支持最强的。"


def _build_system_prompt(domain: str, answer_format: str, *, is_verify: bool = False) -> str:
    """构建注入七维框架+域提示+strictness的system prompt。"""
    strictness = _DOMAIN_STRICTNESS.get(domain, "balanced")
    domain_hint = _DOMAIN_HINTS.get(domain, "")
    base = "你是金融文档合规验证器。只能依据给定证据选择答案，不得编造，不得使用外部知识。"
    if is_verify:
        base += "你正在进行答案自检，请基于证据对每个选项进行证据溯源验证，纠正可能的遗漏或误选。"
    parts = [base, _SEVEN_DIMENSIONS_TEXT]
    if domain_hint:
        parts.append(f"当前领域提示：{domain_hint}")
    parts.append(_build_strictness_rule(strictness, answer_format))
    return "\n\n".join(parts)


def _option_constraint(options: list[str]) -> str:
    """生成选项范围约束语，防止模型编造不存在的选项。"""
    last_letter = OPTION_NAMES[len(options) - 1]
    return f"本题共有{len(options)}个选项（A-{last_letter}），只能从中选择，不得选择未列出的选项。"


def _build_single_messages(
    question: str,
    options: list[str],
    evidence: list[StructuredChunk],
    domain: str = "general",
) -> list[dict[str, str]]:
    """Single-choice prompt: elimination + strongest evidence."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{_format_evidence(evidence)}\n\n"
        "请排除有明确反驳证据的选项，在剩余选项中选择证据支持最强的。"
        f"{_option_constraint(options)}答案必须是单个选项字母。\n"
        '先分析理由，再给答案。输出JSON：{"reasoning":"简述理由","answer":"A"}'
    )
    return [
        {"role": "system", "content": _build_system_prompt(domain, "mcq")},
        {"role": "user", "content": user_prompt},
    ]


def _format_prejudgment(reasoning_hints: list[ReasoningItem]) -> str:
    """将本地逐选项预判格式化为参考锚点（非强制），帮助 Qwen 减少漏选/误选。"""
    if not reasoning_hints:
        return ""
    lines = [f"- 选项{item.option}: 本地初判={item.verdict}" for item in reasoning_hints]
    return (
        "\n【本地初步判定（仅供参考，须以证据为准，可推翻）】\n"
        + "\n".join(lines)
        + "\n"
    )


def _build_multi_messages(
    question: str,
    options: list[str],
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem] | None = None,
    domain: str = "general",
) -> list[dict[str, str]]:
    """Multi-choice prompt: per-option independent judgment."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    prejudgment = _format_prejudgment(reasoning_hints or [])
    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{_format_evidence(evidence)}\n"
        f"{prejudgment}\n"
        "请逐个选项独立判断："
        "有直接证据支持则选；"
        "证据明确排除（含'不得''除外'等否定表述）则不选；"
        "证据部分匹配且无反驳证据的，倾向于选择。"
        "注意：多选题常设2-3个正确选项，过于保守会漏选。"
        f"{_option_constraint(options)}答案必须是多个选项字母按字母顺序拼接。\n"
        '先逐选项分析理由，再给答案。输出JSON：{"reasoning":"简述每选项判定理由","answer":"AC"}'
    )
    return [
        {"role": "system", "content": _build_system_prompt(domain, "multi")},
        {"role": "user", "content": user_prompt},
    ]


def _build_judge_messages(
    question: str,
    options: list[str],
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem] | None = None,
    domain: str = "general",
) -> list[dict[str, str]]:
    """Judge/true-false prompt: negation and exception detection."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    prejudgment = _format_prejudgment(reasoning_hints or [])
    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{_format_evidence(evidence)}\n"
        f"{prejudgment}\n"
        "请特别注意否定表述（不得、不予、除外、不包括）"
        "和例外条款（但...、除...外）可能改变判断方向。"
        f"{_option_constraint(options)}答案必须是单个选项字母。\n"
        '先分析理由，再给答案。输出JSON：{"reasoning":"简述判定理由","answer":"A"}'
    )
    return [
        {"role": "system", "content": _build_system_prompt(domain, "tf")},
        {"role": "user", "content": user_prompt},
    ]


def _build_messages(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem],
    domain: str = "general",
) -> list[dict[str, str]]:
    """Dispatch to format-specific prompt builder.

    reasoning hints 已完全禁用（锚定效应分析确认本地 reasoner 准确率仅 50%，
    错误 refute 预判会导致 LLM 漏选正确选项）。
    """
    # 完全禁用 reasoning hints 注入
    reasoning_hints = []
    if answer_format == "multi":
        return _build_multi_messages(question, options, evidence, reasoning_hints, domain=domain)
    if answer_format == "tf":
        return _build_judge_messages(question, options, evidence, reasoning_hints, domain=domain)
    return _build_single_messages(question, options, evidence, domain=domain)


def _extract_json_object(text: str) -> dict[str, object]:
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model did not return JSON object")
    return json.loads(candidate[start : end + 1])


def _normalize_answer(answer: str, answer_format: str) -> str:
    letters = "".join(sorted(dict.fromkeys(re.findall(r"[A-F]", answer.upper()))))
    if not letters:
        raise ValueError("model answer missing option label")
    if answer_format in ("mcq", "tf") and len(letters) > 1:
        return letters[0]
    return letters


def _extract_answer_from_raw(text: str, answer_format: str) -> str | None:
    """Fallback: try to extract option letters from non-JSON model output."""
    # Try explicit answer patterns first
    raw_patterns = [
        r"答案[是为：:\s]+([A-Fa-f]+)",
        r"正确答案[是为：:\s]+([A-Fa-f]+)",
        r"应选[为是：:\s]+([A-Fa-f]+)",
        r"选择[为了是：:\s]+([A-Fa-f]+)",
        r"选\s*([A-Fa-f]+)\s*[。\.]",
        r"^\s*([A-Fa-f]+)\s*$",
    ]
    for pattern in raw_patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            letters = "".join(sorted(dict.fromkeys(match.group(1).upper())))
            if not letters:
                continue
            if answer_format in ("mcq", "tf") and len(letters) > 1:
                return letters[0]
            return letters
    # Last resort: find all A-F in first 200 chars (answer usually at start)
    head = text[:200]
    letters = "".join(sorted(dict.fromkeys(re.findall(r"[A-F]", head.upper()))))
    if not letters:
        return None
    if answer_format in ("mcq", "tf") and len(letters) > 1:
        return letters[0]
    return letters


def _call_qwen_api(
    messages: list[dict[str, str]],
    answer_format: str,
    api_key: str,
    model_name: str,
    base_url: str,
    *,
    force_thinking: bool | None = None,
) -> QwenAnswer | None:
    """Core API call — shared by first-pass and self-check.

    Returns None when answer cannot be extracted (caller should retry or fallback).
    Includes fallback answer extraction for non-JSON model outputs.

    使用 OpenAI SDK 调用 DashScope 兼容接口。enable_thinking / thinking_budget
    等非标准参数通过 extra_body 传入（赛题教程推荐的传参方式）。

    force_thinking: 显式控制是否开启思考模式。None 时回退到环境变量
    QWEN_ENABLE_THINKING（默认关闭）。调用方可根据题型决定——计算题/比较题
    开启 thinking 有助推理，简单事实查找关闭即可省 token 和时延。
    """
    # enable_thinking 优先级：force_thinking 参数 > 环境变量 > 默认关闭
    if force_thinking is not None:
        enable_thinking = force_thinking
    else:
        enable_thinking = _get_bool_setting("QWEN_ENABLE_THINKING", default=False)
    extra_body: dict[str, object] = {"enable_thinking": enable_thinking}
    if enable_thinking:
        extra_body["thinking_budget"] = 2560  # was 1536; raised for deeper multi-step numerical reasoning

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=f"{base_url}", timeout=90.0)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.01,       # DashScope docs: "Do not set to 0"; 0.01 maximises determinism
        top_p=0.8,              # restrict sampling to high-probability tokens
        seed=42,                # fixed seed for reproducibility
        max_tokens=4096 if answer_format == "multi" else 3072,
        response_format={"type": "json_object"},
        extra_body=extra_body,
    )
    message = completion.choices[0].message.content or ""
    usage_obj = completion.usage
    payload_model = completion.model or model_name

    # Empty content — model returned nothing
    if not message or not message.strip():
        return None

    # Try JSON extraction first
    answer: str | None = None
    reasoning = ""
    try:
        parsed = _extract_json_object(message)
        try:
            answer = _normalize_answer(str(parsed.get("answer", "")), answer_format)
        except ValueError:
            # JSON parsed but answer field invalid — try raw extraction on answer value
            answer = _extract_answer_from_raw(str(parsed.get("answer", "")), answer_format)
        reasoning = str(parsed.get("reasoning", "")).strip()
    except (ValueError, json.JSONDecodeError):
        pass

    # Fallback: extract from raw message text
    if answer is None:
        answer = _extract_answer_from_raw(message, answer_format)
        if answer:
            reasoning = "Qwen 返回非标准JSON，已从文本提取答案。"

    # Still no answer — return None for caller to retry/fallback
    if answer is None:
        return None

    if not reasoning:
        reasoning = "Qwen 已基于检索证据完成选项判断。"

    return QwenAnswer(
        answer=answer,
        reasoning=reasoning,
        prompt_tokens=int(usage_obj.prompt_tokens if usage_obj else 0),
        completion_tokens=int(usage_obj.completion_tokens if usage_obj else 0),
        total_tokens=int(usage_obj.total_tokens if usage_obj else 0),
        model=payload_model,
    )


def _build_simple_messages(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
) -> list[dict[str, str]]:
    """Build a simplified prompt for retry — shorter evidence, no reasoning hints."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    constraint = _option_constraint(options)
    answer_rule = {
        "mcq": f"{constraint}答案必须是单个选项字母，例如 A。",
        "multi": f"{constraint}答案必须是多个选项字母按字母顺序拼接，例如 AC。",
        "tf": f"{constraint}答案必须是单个选项字母。",
    }.get(answer_format, f"{constraint}答案必须是选项字母。")

    # Shorter evidence — top 10 chunks, 500 chars each
    lines: list[str] = []
    for index, chunk in enumerate(evidence[:10], start=1):
        snippet = re.sub(r"\s+", " ", chunk.chunk_text).strip()[:500]
        lines.append(f"{index}. {snippet}")
    evidence_text = "\n".join(lines) if lines else "无候选证据"

    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{evidence_text}\n\n"
        f"请基于证据选择答案。{answer_rule}\n"
        '先分析理由，再给答案。输出JSON：{"reasoning":"...","answer":"A"}'
    )
    return [
        {"role": "system", "content": "你是金融文档阅读理解助手。只依据证据回答。"},
        {"role": "user", "content": user_prompt},
    ]


def answer_with_qwen(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem],
    *,
    force_thinking: bool | None = None,
    domain: str = "general",
) -> QwenAnswer | None:
    status = get_qwen_config_status()
    if not status.enabled:
        return None
    api_key = _get_setting("DASHSCOPE_API_KEY")
    model_name = _get_setting("MODEL_NAME")
    if not api_key or not model_name:
        return None

    base_url = status.base_url

    # Attempt 1: full prompt with reasoning hints
    messages = _build_messages(question, options, answer_format, evidence, reasoning_hints, domain=domain)
    try:
        result = _call_qwen_api(messages, answer_format, api_key, model_name, base_url, force_thinking=force_thinking)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        pass

    # Attempt 2: simplified prompt (no reasoning hints, shorter evidence)
    simple_messages = _build_simple_messages(question, options, answer_format, evidence)
    try:
        result = _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url, force_thinking=force_thinking)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        pass

    # Attempt 3: simplified prompt again (Qwen is non-deterministic — retry may succeed)
    try:
        return _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url, force_thinking=force_thinking)
    except Exception:  # noqa: BLE001
        return None


def verify_with_qwen(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem],
    first_answer: str,
    *,
    domain: str = "general",
) -> QwenAnswer | None:
    """Evidence-tracing self-check.

    For each option, the LLM must quote specific evidence text to justify
    its selection or exclusion. This prevents hallucination and provides
    a truly independent verification.
    """
    status = get_qwen_config_status()
    if not status.enabled:
        return None
    api_key = _get_setting("DASHSCOPE_API_KEY")
    model_name = _get_setting("MODEL_NAME")
    if not api_key or not model_name:
        return None

    base_url = status.base_url
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    constraint = _option_constraint(options)

    if answer_format == "multi":
        example_answer = "AB"
        example_reasoning = "选项A：证据原文...支持；选项B：找不到证据，剔除；..."
        answer_rule = f"{constraint}答案必须是多个选项字母按字母顺序拼接。"
    else:
        example_answer = "A"
        example_reasoning = "选项A：证据原文...支持；选项B：证据不足..."
        answer_rule = f"{constraint}答案必须是单个选项字母。"

    verify_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{_format_evidence(evidence)}\n\n"
        f"第一轮答案：{first_answer}\n\n"
        "【证据溯源自检】\n"
        "请对每个选项进行证据溯源验证：\n"
        "1. 对第一轮选中的选项：从证据中引用原文片段证明该选项正确。找不到直接证据则应剔除。\n"
        "2. 对第一轮未选的选项：检查证据中是否有支持内容。找到支持证据则应纳入。\n"
        "3. 证据明确排除的选项不应选。\n\n"
        f"{answer_rule}\n"
        "先逐选项分析理由，再给最终答案。输出JSON："
        f'{{"reasoning":"{example_reasoning}","answer":"{example_answer}"}}'
    )
    messages = [
        {"role": "system", "content": _build_system_prompt(domain, answer_format, is_verify=True)},
        {"role": "user", "content": verify_prompt},
    ]
    try:
        result = _call_qwen_api(messages, answer_format, api_key, model_name, base_url)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        pass
    # Fallback: simplified prompt retry
    simple_messages = _build_simple_messages(question, options, answer_format, evidence)
    try:
        return _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url)
    except Exception:  # noqa: BLE001
        return None
