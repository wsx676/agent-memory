from __future__ import annotations

import re

from api.models import ReasoningItem, RunQuestionTaskRequest, StructuredChunk
from api.services.planner import activate_rewrite, build_plan, next_rewrite_from_feedback
from api.services.reasoner import reason_options
from api.services.retrieval_evaluator import evaluate_retrieval_quality
from api.services.retriever import rank_chunks, rank_documents

OPTION_NAMES = ("A", "B", "C", "D", "E", "F")


def _resolve_doc_ids(
    payload: RunQuestionTaskRequest,
    *,
    available_doc_ids: set[str],
    structured_chunks: list[StructuredChunk],
    plan: dict[str, object],
    previous_doc_ids: list[str],
) -> list[str]:
    if payload.mode == "A":
        return [doc_id for doc_id in (payload.doc_ids or []) if doc_id in available_doc_ids]

    ranked_docs = [doc_id for doc_id in rank_documents(payload.question, structured_chunks, plan) if doc_id in available_doc_ids]
    if plan.get("requires_multi_doc"):
        combined = []
        for doc_id in [*previous_doc_ids, *ranked_docs]:
            if doc_id not in combined:
                combined.append(doc_id)
        return combined[:8]
    return ranked_docs


def _init_ledger(options: list[str]) -> dict[str, object]:
    """初始化跨轮结构化账本。

    账本跟踪每选项的收敛状态，驱动第二轮起的定向补检索：
    - option_status: 每选项的当前 verdict（unknown→support/refute/insufficient）
    - confirmed_chunks: 已确认选项的证据，跨轮累积不丢失
    - gaps: insufficient 选项的文本，用于生成定向 query
    """
    return {
        "option_status": {name: "unknown" for name in OPTION_NAMES[: len(options)]},
        "confirmed_results": {},  # option_name → ReasoningItem（已确认的）
        "confirmed_chunks": [],   # 已确认选项的 evidence chunks（跨轮累积）
        "all_candidate_chunks": [],  # 所有轮次的候选 chunks（去重累积）
        "gaps": [],  # insufficient 选项的文本列表
    }


def _update_ledger(
    ledger: dict[str, object],
    reasoning_results: list[ReasoningItem],
    evidence_map: dict[str, list[StructuredChunk]],
    candidate_chunks: list[StructuredChunk],
) -> None:
    """每轮结束后更新账本：固化已确认选项，提取缺口。"""
    option_status = dict(ledger["option_status"])
    confirmed_results = dict(ledger["confirmed_results"])
    gaps: list[str] = []

    for result in reasoning_results:
        option_status[result.option] = result.verdict
        if result.verdict in ("support", "refute"):
            # 固化已确认选项的推理结果和证据
            confirmed_results[result.option] = result
        elif result.verdict == "insufficient":
            # 记录缺口——insufficient 选项的文本将驱动下一轮 query
            gaps.append(result.option)

    ledger["option_status"] = option_status
    ledger["confirmed_results"] = confirmed_results
    ledger["gaps"] = gaps

    # 累积候选 chunks（去重）
    existing_ids = {c.chunk_id for c in ledger["all_candidate_chunks"]}
    for chunk in candidate_chunks:
        if chunk.chunk_id not in existing_ids:
            ledger["all_candidate_chunks"].append(chunk)
            existing_ids.add(chunk.chunk_id)

    # 累积已确认选项的证据
    confirmed_ids = {c.chunk_id for c in ledger["confirmed_chunks"]}
    for option_name, result in confirmed_results.items():
        for chunk in evidence_map.get(option_name, []):
            if chunk.chunk_id not in confirmed_ids:
                ledger["confirmed_chunks"].append(chunk)
                confirmed_ids.add(chunk.chunk_id)


def _build_gap_driven_plan(
    base_plan: dict[str, object],
    ledger: dict[str, object],
    options: list[str],
    evaluation: dict[str, object],
) -> dict[str, object]:
    """根据账本中的 gaps 生成定向补检索计划。

    与原 next_rewrite_from_feedback 的区别：
    - option_focus 由 ledger 中实际的 insufficient 选项驱动（而非泛化策略）
    - query_hints 注入 insufficient 选项的文本片段，让检索直接命中缺口
    """
    action = str(evaluation.get("next_action", "accept") or "accept")
    if action == "accept":
        return activate_rewrite(base_plan, "base")

    # gaps 是 insufficient 选项的字母名
    gap_options = [str(g) for g in ledger.get("gaps", [])]
    # 提取 insufficient 选项的文本作为定向检索焦点
    option_focus = [
        options[i]
        for i, name in enumerate(OPTION_NAMES[: len(options)])
        if name in gap_options
    ]

    # 仍用 next_rewrite_from_feedback 选择 rewrite 策略，但注入 gap-driven 的 option_focus
    plan = next_rewrite_from_feedback(base_plan, evaluation, options)

    # 如果有 gap-driven 的选项焦点，注入到 plan 中让检索器优先搜这些选项
    if option_focus:
        # 合并而非替换——保留策略 hints，追加 gap 选项文本
        existing_hints = list(plan.get("query_hints", []))
        # 将 insufficient 选项的核心内容作为额外 hints
        for opt_text in option_focus:
            # 取选项前 60 字作为定向 hint
            existing_hints.append(opt_text[:60])
        plan["query_hints"] = existing_hints
        plan["option_focus"] = option_focus

    return plan


# reasoning 文本里嵌入的置信度标记，形如"（置信度85%）"
_CONFIDENCE_PATTERN = re.compile(r"置信度\s*(\d+(?:\.\d+)?)\s*%")

# 允许翻转已确认结论所需的最小置信度增量：新结论必须显著更强才覆盖旧结论，
# 防止低置信噪声引起轮次间震荡。
_FLIP_CONFIDENCE_MARGIN = 0.15


def _extract_confidence(item: ReasoningItem) -> float:
    """从 reasoning 文本解析置信度（0~1）。

    reasoner 目前只把置信度写进 reasoning 文本（如"（置信度85%）"），
    未落到 ReasoningItem 字段，因此这里用正则解析。解析失败返回 0.0。
    """
    match = _CONFIDENCE_PATTERN.search(item.reasoning or "")
    if not match:
        return 0.0
    try:
        return float(match.group(1)) / 100.0
    except ValueError:
        return 0.0


def _merge_reasoning_results(
    confirmed: dict[str, ReasoningItem],
    new_results: list[ReasoningItem],
    options: list[str],
) -> list[ReasoningItem]:
    """合并已确认的推理结果和新一轮的推理结果。

    收敛策略（在单调收敛基础上允许有条件纠错）：
    - 原先 insufficient/unknown 的选项：直接接受新结果。
    - 已确认为 support/refute 的选项：默认保留旧结论；但当新一轮同样给出
      确定性结论（support/refute）且置信度显著更高（增量 >= _FLIP_CONFIDENCE_MARGIN）
      时，用新结论覆盖——避免首轮因证据不全的误判被永久固化，同时用 margin
      阈值抑制低置信噪声导致的来回翻转。
    """
    merged: dict[str, ReasoningItem] = {}
    # 先放入已确认的
    for name, item in confirmed.items():
        merged[name] = item
    # 新结果按规则覆盖
    for result in new_results:
        existing = merged.get(result.option)
        if existing is None or existing.verdict not in ("support", "refute"):
            # 未确认项：直接接受新结果
            merged[result.option] = result
            continue
        # 已确认项：仅当新结论也是确定性且置信度显著更高时才翻转
        if result.verdict in ("support", "refute"):
            new_conf = _extract_confidence(result)
            old_conf = _extract_confidence(existing)
            if new_conf - old_conf >= _FLIP_CONFIDENCE_MARGIN:
                merged[result.option] = result
    # 按选项字母排序输出
    return [merged[name] for name in OPTION_NAMES[: len(options)] if name in merged]


def run_retrieval_loop(
    payload: RunQuestionTaskRequest,
    *,
    available_doc_ids: set[str],
    structured_chunks: list[StructuredChunk],
) -> dict[str, object]:
    base_plan = build_plan(payload)
    current_plan = activate_rewrite(base_plan, "base")
    previous_doc_ids: list[str] = []
    loop_logs: list[str] = []
    best_round: dict[str, object] | None = None
    max_iterations = int(base_plan.get("max_iterations", 3) or 3)

    # 初始化跨轮账本
    ledger = _init_ledger(payload.options)

    for iteration in range(1, max_iterations + 1):
        current_plan["iteration"] = iteration
        current_doc_ids = _resolve_doc_ids(
            payload,
            available_doc_ids=available_doc_ids,
            structured_chunks=structured_chunks,
            plan=current_plan,
            previous_doc_ids=previous_doc_ids,
        )
        previous_doc_ids = current_doc_ids

        # 检索：第二轮起使用累积的全部候选 + 新检索结果
        candidate_chunks = rank_chunks(
            payload.question,
            payload.options,
            structured_chunks,
            current_doc_ids or None,
            current_plan,
        )

        # 第二轮起：合并累积的已确认证据，让 reasoner 能看到跨轮证据
        if iteration > 1 and ledger["confirmed_chunks"]:
            existing_ids = {c.chunk_id for c in candidate_chunks}
            for chunk in ledger["confirmed_chunks"]:
                if chunk.chunk_id not in existing_ids:
                    candidate_chunks.extend([chunk])
                    existing_ids.add(chunk.chunk_id)

        reasoning_results, evidence_map = reason_options(
            payload.options, candidate_chunks, payload.question, payload.answer_format,
        )

        # 更新账本（跨轮累积）
        _update_ledger(ledger, reasoning_results, evidence_map, candidate_chunks)

        # 合并推理结果：已确认的保留，只接受新确认的
        if iteration > 1:
            reasoning_results = _merge_reasoning_results(
                ledger["confirmed_results"], reasoning_results, payload.options,
            )

        evaluation = evaluate_retrieval_quality(
            question=payload.question,
            options=payload.options,
            plan=current_plan,
            candidate_chunks=candidate_chunks,
            reasoning_results=reasoning_results,
            candidate_doc_ids=current_doc_ids,
        )
        round_result = {
            "plan": dict(current_plan),
            "doc_ids": list(current_doc_ids),
            "candidate_chunks": candidate_chunks,
            "reasoning_results": reasoning_results,
            "evidence_map": evidence_map,
            "evaluation": evaluation,
        }
        if best_round is None or float(evaluation["quality_score"]) >= float(best_round["evaluation"]["quality_score"]):
            best_round = round_result

        loop_logs.extend(
            [
                f"react_round={iteration}",
                f"react_rewrite={current_plan.get('active_rewrite_name', 'base')}",
                f"react_doc_count={len(current_doc_ids)}",
                f"react_chunk_count={len(candidate_chunks)}",
                f"react_quality={evaluation['quality_score']}",
                f"react_quality_label={evaluation['quality_label']}",
                f"react_decision={evaluation['decision']}",
                f"react_next_action={evaluation['next_action']}",
                f"react_failures={','.join(evaluation['failure_reasons']) if evaluation['failure_reasons'] else 'none'}",
                f"react_gaps={','.join(ledger['gaps']) if ledger['gaps'] else 'none'}",
                f"react_confirmed={','.join(k for k,v in ledger['option_status'].items() if v in ('support','refute')) or 'none'}",
            ]
        )

        if evaluation["decision"] == "accept" or iteration >= max_iterations:
            chosen = round_result if evaluation["decision"] == "accept" else best_round or round_result
            return {
                **chosen,
                "base_plan": base_plan,
                "loop_logs": loop_logs,
            }

        # 由 gaps 驱动下一轮的 query 改写
        current_plan = _build_gap_driven_plan(base_plan, ledger, payload.options, evaluation)

    chosen = best_round or {
        "plan": current_plan,
        "doc_ids": [],
        "candidate_chunks": [],
        "reasoning_results": [],
        "evidence_map": {},
        "evaluation": {"quality_score": 0.0, "quality_label": "low", "decision": "retry", "next_action": "accept"},
    }
    return {
        **chosen,
        "base_plan": base_plan,
        "loop_logs": loop_logs,
    }
