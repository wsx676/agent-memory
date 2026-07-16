from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from api.models import ReasoningItem, StructuredChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
OPTION_NAMES = ("A", "B", "C", "D", "E", "F")


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


def _format_evidence(chunks: list[StructuredChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks[:10], start=1):
        location = f"doc={chunk.doc_id}, page={chunk.page_no}"
        if chunk.clause_no:
            location += f", clause={chunk.clause_no}"
        if chunk.section_path:
            location += f", section={chunk.section_path}"
        snippet = re.sub(r"\s+", " ", chunk.chunk_text).strip()[:600]
        lines.append(f"{index}. [{location}] {snippet}")
    return "\n".join(lines) if lines else "无候选证据"


def _format_reasoning_hints(results: list[ReasoningItem]) -> str:
    """Format per-option analysis with verdict and key evidence summary."""
    lines = []
    for item in results:
        lines.append(f"- 选项{item.option}: {item.verdict}；{item.reasoning[:200]}")
    return "\n".join(lines) if lines else "- 无本地推理提示"


def _build_single_messages(
    question: str,
    options: list[str],
    evidence: list[StructuredChunk],
) -> list[dict[str, str]]:
    """Single-choice prompt: elimination + strongest evidence."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{_format_evidence(evidence)}\n\n"
        "请排除有明确反驳证据的选项，在剩余选项中选择证据支持最强的。"
        "答案必须是单个选项字母。\n"
        '输出JSON：{"answer":"A","reasoning":"简述理由"}'
    )
    return [
        {"role": "system", "content": "你是金融文档阅读理解助手。只能依据给定证据选择最佳答案，不得编造。"},
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
) -> list[dict[str, str]]:
    """Multi-choice prompt: per-option independent judgment."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    prejudgment = _format_prejudgment(reasoning_hints or [])
    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{_format_evidence(evidence)}\n"
        f"{prejudgment}\n"
        "请逐个选项独立判断：有直接证据支持则选，"
        "证据明确排除（含'不得''除外'等否定表述）则不选，"
        "无法从证据中确认的不选。"
        "答案必须是多个选项字母按字母顺序拼接。\n"
        '输出JSON：{"answer":"AC","reasoning":"简述每选项判定理由"}'
    )
    return [
        {"role": "system", "content": "你是金融文档阅读理解助手。只能依据给定证据选择所有正确答案，不得编造。"},
        {"role": "user", "content": user_prompt},
    ]


def _build_judge_messages(
    question: str,
    options: list[str],
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem] | None = None,
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
        "答案必须是单个选项字母。\n"
        '输出JSON：{"answer":"A","reasoning":"简述判定理由"}'
    )
    return [
        {"role": "system", "content": "你是金融文档阅读理解助手。只能依据给定证据判断对错，不得编造。"},
        {"role": "user", "content": user_prompt},
    ]


def _build_messages(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem],
) -> list[dict[str, str]]:
    """Dispatch to format-specific prompt builder.

    多选与判断题注入本地逐选项预判作为参考锚点（最受漏选/否定误判影响）；
    单选题保持精简，不注入以控制 token 与避免本地误判带偏。
    """
    if answer_format == "multi":
        return _build_multi_messages(question, options, evidence, reasoning_hints)
    if answer_format == "judge":
        return _build_judge_messages(question, options, evidence, reasoning_hints)
    return _build_single_messages(question, options, evidence)


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
    if answer_format == "single" and len(letters) > 1:
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
            if answer_format == "single" and len(letters) > 1:
                return letters[0]
            return letters
    # Last resort: find all A-F in first 200 chars (answer usually at start)
    head = text[:200]
    letters = "".join(sorted(dict.fromkeys(re.findall(r"[A-F]", head.upper()))))
    if not letters:
        return None
    if answer_format == "single" and len(letters) > 1:
        return letters[0]
    return letters


def _call_qwen_api(
    messages: list[dict[str, str]],
    answer_format: str,
    api_key: str,
    model_name: str,
    base_url: str,
) -> QwenAnswer | None:
    """Core API call — shared by first-pass and self-check.

    Returns None when answer cannot be extracted (caller should retry or fallback).
    Includes fallback answer extraction for non-JSON model outputs.
    """
    # enable_thinking 默认关闭：P2v3 验证过 Qwen3 思考模式会产生 500+ 内部 token、
    # 非确定性输出与偶发空/截断响应，关闭后 100 题全部首次成功、延迟降 71%。
    # 保留 QWEN_ENABLE_THINKING 开关以便 A/B 对比，但默认回到已验证的最优配置。
    enable_thinking = _get_bool_setting("QWEN_ENABLE_THINKING", default=False)
    request_body: dict[str, object] = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 4096 if answer_format == "multi" else 3072,
        "response_format": {"type": "json_object"},
        "messages": messages,
        "enable_thinking": enable_thinking,
    }
    if enable_thinking:
        request_body["thinking_budget"] = 2560
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=90.0,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0]["message"]["content"]
    usage = payload.get("usage", {})

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
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
        model=str(payload.get("model", model_name)),
    )


def _build_simple_messages(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
) -> list[dict[str, str]]:
    """Build a simplified prompt for retry — shorter evidence, no reasoning hints."""
    option_lines = [f"{name}. {option}" for name, option in zip(OPTION_NAMES, options)]
    answer_rule = {
        "single": "答案必须是单个选项字母，例如 A。",
        "multi": "答案必须是多个选项字母按字母顺序拼接，例如 AC。",
        "judge": "答案必须是单个选项字母。",
    }.get(answer_format, "答案必须是选项字母。")

    # Shorter evidence — top 5 chunks, 300 chars each
    lines: list[str] = []
    for index, chunk in enumerate(evidence[:5], start=1):
        snippet = re.sub(r"\s+", " ", chunk.chunk_text).strip()[:300]
        lines.append(f"{index}. {snippet}")
    evidence_text = "\n".join(lines) if lines else "无候选证据"

    user_prompt = (
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"证据：\n{evidence_text}\n\n"
        f"请基于证据选择答案。{answer_rule}\n"
        '输出JSON：{"answer":"A","reasoning":"..."}'
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
    messages = _build_messages(question, options, answer_format, evidence, reasoning_hints)
    try:
        result = _call_qwen_api(messages, answer_format, api_key, model_name, base_url)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        pass

    # Attempt 2: simplified prompt (no reasoning hints, shorter evidence)
    simple_messages = _build_simple_messages(question, options, answer_format, evidence)
    try:
        result = _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url)
        if result is not None:
            return result
    except Exception:  # noqa: BLE001
        pass

    # Attempt 3: simplified prompt again (Qwen is non-deterministic — retry may succeed)
    try:
        return _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url)
    except Exception:  # noqa: BLE001
        return None


def verify_with_qwen(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem],
    first_answer: str,
) -> QwenAnswer | None:
    """Evidence-tracing self-check — independent of reasoning hints.

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
        "输出JSON："
        '{"answer":"AB","reasoning":"选项A：证据原文...支持；选项B：找不到证据，剔除；..."}'
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是金融文档阅读理解助手，正在进行答案自检。"
                "请基于证据对每个选项进行证据溯源验证，纠正可能的遗漏或误选。"
                "只能依据给定证据做选择，不得编造。"
            ),
        },
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
