"""res_a 5道错题全链路诊断脚本。

对每道错题输出完整运行轨迹：
1. 检索阶段：轮次、重写策略、质量评分、候选chunk列表
2. 本地推理：每选项verdict/reasoning
3. LLM答题：LLM answer+reasoning、重试次数
4. self-check：是否触发、前后答案对比、verify reasoning
5. 与gold答案对比

输出到 .tmp_diagnostic_res.txt（UTF-8）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk, ReasoningItem
from api.services.qwen_client import (
    QwenAnswer, answer_with_qwen, verify_with_qwen, get_qwen_config_status,
    _format_evidence, _build_messages, _call_qwen_api, _get_setting,
    _build_simple_messages, OPTION_NAMES,
)
from api.services.formatter import needs_self_check
from api.services.retrieval_loop import run_retrieval_loop

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_FILE = PROJECT_ROOT / ".tmp_diagnostic_res.txt"

ERROR_QIDS = ["res_a_001", "res_a_002", "res_a_018", "res_a_019", "res_a_020"]

GOLD_ANSWERS = {
    "res_a_001": "ABC",
    "res_a_002": "AC",
    "res_a_018": "A",
    "res_a_019": "ABD",
    "res_a_020": "ABD",
}

OPTION_KEYS = ("A", "B", "C", "D", "E", "F")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answer_with_qwen_traced(
    question: str,
    options: list[str],
    answer_format: str,
    evidence: list[StructuredChunk],
    reasoning_hints: list[ReasoningItem],
    force_thinking: bool | None = None,
) -> tuple[QwenAnswer | None, dict]:
    trace: dict = {
        "attempts": [],
        "formatted_evidence": _format_evidence(evidence),
        "evidence_char_count": len(_format_evidence(evidence)),
    }

    status = get_qwen_config_status()
    if not status.enabled:
        return None, trace

    api_key = _get_setting("DASHSCOPE_API_KEY")
    model_name = _get_setting("MODEL_NAME")
    if not api_key or not model_name:
        return None, trace

    base_url = status.base_url

    # Attempt 1: full prompt
    messages = _build_messages(question, options, answer_format, evidence, reasoning_hints)
    try:
        result = _call_qwen_api(messages, answer_format, api_key, model_name, base_url, force_thinking=force_thinking)
        trace["attempts"].append({"attempt": 1, "type": "full_prompt", "success": result is not None})
        if result is not None:
            trace["used_attempt"] = 1
            return result, trace
    except Exception as e:
        trace["attempts"].append({"attempt": 1, "type": "full_prompt", "success": False, "error": str(e)})

    # Attempt 2: simplified prompt
    simple_messages = _build_simple_messages(question, options, answer_format, evidence)
    try:
        result = _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url, force_thinking=force_thinking)
        trace["attempts"].append({"attempt": 2, "type": "simple_prompt", "success": result is not None})
        if result is not None:
            trace["used_attempt"] = 2
            return result, trace
    except Exception as e:
        trace["attempts"].append({"attempt": 2, "type": "simple_prompt", "success": False, "error": str(e)})

    # Attempt 3: simplified prompt again
    try:
        result = _call_qwen_api(simple_messages, answer_format, api_key, model_name, base_url, force_thinking=force_thinking)
        trace["attempts"].append({"attempt": 3, "type": "simple_retry", "success": result is not None})
        if result is not None:
            trace["used_attempt"] = 3
            return result, trace
    except Exception as e:
        trace["attempts"].append({"attempt": 3, "type": "simple_retry", "success": False, "error": str(e)})

    trace["used_attempt"] = 0
    return None, trace


def main() -> None:
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks_data = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / "chunks_merged.jsonl")]

    questions_file = QUESTIONS_DIR / "research_questions.json"
    all_questions = json.loads(questions_file.read_text(encoding="utf-8"))

    out: list[str] = []
    out.append("=" * 100)
    out.append("res_a 5道错题全链路诊断轨迹")
    out.append("=" * 100)

    for raw in all_questions:
        qid = raw["qid"]
        if qid not in ERROR_QIDS:
            continue

        gold = GOLD_ANSWERS[qid]
        options_dict = raw.get("options", {})
        options = [options_dict[k] for k in OPTION_KEYS if k in options_dict]
        answer_format = raw["answer_format"]
        doc_ids = raw.get("doc_ids", [])

        started = time.perf_counter()

        out.append(f"\n\n{'#' * 100}")
        out.append(f"# 题目: {qid}  Gold: {gold}  题型: {answer_format}")
        out.append(f"# 问题: {raw['question'][:200]}")
        for k in OPTION_KEYS:
            if k in options_dict:
                out.append(f"#   选项{k}: {options_dict[k][:120]}")
        out.append(f"{'#' * 100}")

        # ── 1. 检索阶段 ──
        out.append(f"\n{'=' * 80}")
        out.append("1. 检索阶段 (run_retrieval_loop)")
        out.append(f"{'=' * 80}")

        request = RunQuestionTaskRequest(
            mode="A", qid=qid, question=raw["question"],
            options=options, answerFormat=answer_format, docIds=doc_ids,
        )
        retrieval = run_retrieval_loop(request, available_doc_ids=available_doc_ids, structured_chunks=chunks_data)

        candidate_chunks = retrieval["candidate_chunks"]
        reasoning_results = retrieval["reasoning_results"]
        evidence_map = retrieval["evidence_map"]
        evaluation = retrieval["evaluation"]
        loop_logs = retrieval.get("loop_logs", [])
        plan = retrieval.get("plan", {})

        out.append(f"  domain={plan.get('domain', '?')}, question_type={plan.get('question_type', '?')}")
        out.append(f"  max_iterations={plan.get('max_iterations', 3)}")
        out.append(f"  active_rewrite={plan.get('active_rewrite_name', 'base')}")
        out.append(f"  doc_ids: {retrieval['doc_ids']}")
        out.append(f"  candidate_chunks: {len(candidate_chunks)}")

        out.append(f"\n  --- 检索循环日志 ---")
        for log in loop_logs:
            out.append(f"  {log}")

        out.append(f"\n  --- 质量评估 ---")
        out.append(f"  quality_score={evaluation.get('quality_score', 0):.4f}")
        out.append(f"  quality_label={evaluation.get('quality_label', '?')}")
        out.append(f"  decision={evaluation.get('decision', '?')}")
        out.append(f"  failure_reasons={evaluation.get('failure_reasons', [])}")
        out.append(f"  insufficient_options={evaluation.get('insufficient_options', [])}")

        out.append(f"\n  --- 候选chunk列表 ---")
        for i, c in enumerate(candidate_chunks):
            snippet = c.chunk_text[:80].replace("\n", " ")
            out.append(f"  [{i:2d}] {c.chunk_id} (p{c.page_no}, {c.chunk_type}, sec={c.section_path[:30]}) {snippet}")

        # ── 2. 本地推理 ──
        out.append(f"\n{'=' * 80}")
        out.append("2. 本地推理 (reasoner)")
        out.append(f"{'=' * 80}")

        for r in reasoning_results:
            ev_chunks = evidence_map.get(r.option, [])
            out.append(f"  选项{r.option}: verdict={r.verdict}")
            out.append(f"    reasoning: {r.reasoning[:200]}")
            out.append(f"    evidence_chunks: {[c.chunk_id for c in ev_chunks]}")

        # ── 3. LLM答题 ──
        out.append(f"\n{'=' * 80}")
        out.append("3. LLM答题 (answer_with_qwen)")
        out.append(f"{'=' * 80}")

        qwen_result, qwen_trace = answer_with_qwen_traced(
            question=request.question,
            options=request.options,
            answer_format=request.answer_format,
            evidence=list(candidate_chunks),
            reasoning_hints=list(reasoning_results),
            force_thinking=True,
        )

        out.append(f"  evidence_char_count: {qwen_trace['evidence_char_count']}")
        out.append(f"  used_attempt: {qwen_trace.get('used_attempt', 0)}")
        out.append(f"  attempts: {qwen_trace['attempts']}")

        llm_answer = qwen_result.answer if qwen_result else "A"
        llm_before_selfcheck = llm_answer

        if qwen_result:
            out.append(f"\n  --- LLM回答 ---")
            out.append(f"  answer: {llm_answer}")
            out.append(f"  tokens: prompt={qwen_result.prompt_tokens}, completion={qwen_result.completion_tokens}, total={qwen_result.total_tokens}")
            out.append(f"\n  --- LLM推理文本 ---")
            out.append(f"  {qwen_result.reasoning}")
        else:
            out.append(f"  LLM返回None，使用默认答案A")

        # ── 4. self-check ──
        out.append(f"\n{'=' * 80}")
        out.append("4. 证据溯源自检 (verify_with_qwen)")
        out.append(f"{'=' * 80}")

        do_self_check = needs_self_check(request.answer_format, list(reasoning_results), llm_answer)
        out.append(f"  needs_self_check: {do_self_check}")

        if request.answer_format == "multi":
            out.append(f"  trigger_reason: (multi题型总是触发)")
        elif not any(r.verdict == "support" for r in reasoning_results):
            out.append(f"  trigger_reason: (无support信号)")
        else:
            out.append(f"  trigger_reason: (不触发)")

        if do_self_check and qwen_result:
            try:
                verified = verify_with_qwen(
                    question=request.question,
                    options=request.options,
                    answer_format=request.answer_format,
                    evidence=list(candidate_chunks),
                    reasoning_hints=list(reasoning_results),
                    first_answer=llm_answer,
                )
                if verified:
                    out.append(f"\n  --- 自检结果 ---")
                    out.append(f"  verify_answer: {verified.answer}")
                    out.append(f"  answer_changed: {llm_answer != verified.answer}")
                    out.append(f"\n  --- 自检推理文本 ---")
                    out.append(f"  {verified.reasoning}")

                    if verified.answer != llm_answer:
                        out.append(f"\n  ⚠️ 自检改变了答案: {llm_answer} -> {verified.answer}")
                    llm_answer = verified.answer
                else:
                    out.append(f"  verify返回None")
            except Exception as e:
                out.append(f"  自检异常: {e}")
        else:
            out.append(f"  未触发自检")

        # ── 5. 最终对比 ──
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        out.append(f"\n{'=' * 80}")
        out.append("5. 最终对比")
        out.append(f"{'=' * 80}")
        out.append(f"  gold:           {gold}")
        out.append(f"  LLM首轮:        {llm_before_selfcheck}  {'✅' if llm_before_selfcheck == gold else '❌'}")
        out.append(f"  自检后(最终):   {llm_answer}  {'✅' if llm_answer == gold else '❌'}")
        out.append(f"  自检影响:       {'改了答案' if llm_answer != llm_before_selfcheck else '未改'}")
        out.append(f"  耗时: {duration_ms:.0f}ms")

        if llm_answer != gold:
            gold_set = set(gold)
            llm_set = set(llm_answer)
            missing = gold_set - llm_set
            extra = llm_set - gold_set
            out.append(f"\n  --- 错误分析 ---")
            if missing:
                out.append(f"  漏选: {sorted(missing)}")
            if extra:
                out.append(f"  误选: {sorted(extra)}")

    OUTPUT_FILE.write_text("\n".join(out), encoding="utf-8")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
