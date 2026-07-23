"""赛题标准提交入口 — 一键运行 A/B 榜问答并生成提交文件。

用法:
    python run.py --split A --workers 8
    python run.py --split A --workers 8 --output output/
    python run.py --split A --limit 5            # 只跑前 5 题（调试用）

输出:
    output/submission_a.csv   # 赛题提交格式（含 summary 行）
    output/results_a.jsonl    # 详细结果日志
    output/evidence_a.json    # 证据溯源

环境变量（.env）:
    ENABLE_QWEN_LLM=1
    DASHSCOPE_API_KEY=sk-...
    MODEL_NAME=qwen-plus
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.formatter import build_evidence_items, needs_self_check
from api.services.qwen_client import QwenAnswer, answer_with_qwen, get_qwen_config_status, verify_with_qwen
from api.services.retrieval_loop import run_retrieval_loop

# 题型 → 是否开启 thinking 的映射。
# 全题型开启 thinking：数值比较/判断推理均需分步思考，关闭会导致 tf 题漏判。
THINKING_QUESTION_TYPES = {"calculation", "comparison", "tf", "multi_select", "fact_lookup", "clause_lookup"}


# ── 路径常量 ────────────────────────────────────────────────────────
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"

VALID_ANSWER_FORMATS = ("mcq", "multi", "tf")
_ANSWER_FORMAT_ALIASES = {
    "单选": "mcq", "单选题": "mcq",
    "多选": "multi", "多选题": "multi",
    "判断": "tf", "判断题": "tf",
}
OPTION_KEYS = ("A", "B", "C", "D", "E", "F")


def _resolve_answer_format(raw: str) -> str | None:
    """Resolve answer_format to canonical mcq/multi/tf, with Chinese alias support."""
    lower = raw.lower()
    if lower in VALID_ANSWER_FORMATS:
        return lower
    return _ANSWER_FORMAT_ALIASES.get(lower)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_options(raw: Any) -> list[str] | None:
    if isinstance(raw, dict):
        ordered = [raw[k] for k in OPTION_KEYS if k in raw]
        return ordered or None
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    return None


def load_cases() -> list[dict[str, Any]]:
    """加载全部题目（100 题，5 域 × 20 题）。"""
    cases: list[dict[str, Any]] = []
    for file_path in sorted(QUESTIONS_DIR.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for raw in payload:
            options = normalize_options(raw.get("options"))
            answer_format = _resolve_answer_format(str(raw.get("answer_format", "")))
            if not options or not answer_format:
                continue
            cases.append({
                "qid": str(raw["qid"]),
                "domain": str(raw.get("domain", "unknown")),
                "question": str(raw["question"]),
                "options": options,
                "answer_format": answer_format,
                "doc_ids": [str(item) for item in raw.get("doc_ids", [])],
            })
    return cases


def process_one_case(
    case: dict[str, Any],
    available_doc_ids: set[str],
    chunks: list[StructuredChunk],
) -> dict[str, Any]:
    """处理单道题目：检索 → 推理 → Qwen 答题 → 自检。"""
    started = time.perf_counter()
    request = RunQuestionTaskRequest(
        mode="A",
        qid=case["qid"],
        question=case["question"],
        options=case["options"],
        answerFormat=case["answer_format"],
        docIds=case["doc_ids"],
    )
    retrieval = run_retrieval_loop(request, available_doc_ids=available_doc_ids, structured_chunks=chunks)
    evidence_items = build_evidence_items(retrieval["reasoning_results"], retrieval["evidence_map"])

    # 全题型开启 thinking：数值比较/判断推理均需分步思考
    use_thinking = True

    qwen_error = None
    qwen_result: QwenAnswer | None = None
    try:
        qwen_result = answer_with_qwen(
            question=request.question,
            options=request.options,
            answer_format=request.answer_format,
            evidence=list(retrieval["candidate_chunks"]),
            reasoning_hints=list(retrieval["reasoning_results"]),
            force_thinking=use_thinking,
            domain=case["domain"],
        )
    except Exception as exc:  # noqa: BLE001
        qwen_error = f"{type(exc).__name__}: {exc}"

    final_answer = qwen_result.answer if qwen_result is not None else "A"

    # 低置信度答案自检
    self_checked = False
    if qwen_result is not None and needs_self_check(request.answer_format, list(retrieval["reasoning_results"]), final_answer):
        self_checked = True
        try:
            verified = verify_with_qwen(
                question=request.question,
                options=request.options,
                answer_format=request.answer_format,
                evidence=list(retrieval["candidate_chunks"]),
                reasoning_hints=list(retrieval["reasoning_results"]),
                first_answer=final_answer,
                domain=case["domain"],
            )
            if verified is not None:
                final_answer = verified.answer
                qwen_result = QwenAnswer(
                    answer=final_answer,
                    reasoning=verified.reasoning,
                    prompt_tokens=qwen_result.prompt_tokens + verified.prompt_tokens,
                    completion_tokens=qwen_result.completion_tokens + verified.completion_tokens,
                    total_tokens=qwen_result.total_tokens + verified.total_tokens,
                    model=verified.model,
                )
        except Exception:  # noqa: BLE001
            pass

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "qid": case["qid"],
        "domain": case["domain"],
        "answer": final_answer,
        "prompt_tokens": int(qwen_result.prompt_tokens) if qwen_result is not None else 0,
        "completion_tokens": int(qwen_result.completion_tokens) if qwen_result is not None else 0,
        "total_tokens": int(qwen_result.total_tokens) if qwen_result is not None else 0,
        "duration_ms": duration_ms,
        "evidence": [item.model_dump(by_alias=True) for item in evidence_items],
        "error": qwen_error,
    }


def normalize_answer(answer: str) -> str:
    """答案标准化：提取 A-F 大写字母，排序去重，空答案兜底为 A。"""
    letters = "".join(sorted(set(re.findall(r"[A-F]", (answer or "").upper()))))
    return letters or "A"


def write_submission_csv(path: Path, results: list[dict[str, Any]]) -> None:
    """生成赛题标准 submission CSV（含 summary 汇总行）。"""
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total_all = sum(r["total_tokens"] for r in results)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", total_prompt, total_completion, total_all])
        for r in results:
            writer.writerow([r["qid"], normalize_answer(r["answer"]), r["prompt_tokens"], r["completion_tokens"], r["total_tokens"]])


def write_results_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_evidence_json(path: Path, results: list[dict[str, Any]]) -> None:
    evidence_list = []
    for r in results:
        evidence_list.append({
            "qid": r["qid"],
            "domain": r["domain"],
            "answer": normalize_answer(r["answer"]),
            "evidence_retrieval": r["evidence"],
            "token_usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
            },
        })
    path.write_text(json.dumps(evidence_list, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="赛题四标准提交入口")
    parser.add_argument("--split", type=str, default="A", choices=["A", "B"], help="A 榜或 B 榜（默认 A）")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（百炼 API 限制约 10，建议 8）")
    parser.add_argument("--output", type=str, default="output", help="输出目录（默认 output/）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 题（0=全部，调试用）")
    parser.add_argument("--chunks-file", type=str, default="chunks_merged.jsonl", help="切片文件名")
    args = parser.parse_args()

    split = args.split.lower()
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查 Qwen 配置
    qwen_status = get_qwen_config_status()
    if not qwen_status.enabled:
        missing = ",".join(qwen_status.missing_settings) or "none"
        print(f"⚠️  Qwen 未启用: missing={missing} reason={qwen_status.disable_reason}")
        print("   将使用纯规则推理（准确率会下降）。请在 .env 中配置 ENABLE_QWEN_LLM/DASHSCOPE_API_KEY/MODEL_NAME")
    else:
        print(f"✓ Qwen 已启用: model={qwen_status.requested_model}")

    # 加载数据
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / args.chunks_file)]
    print(f"加载切片: {len(chunks)} 条, 覆盖 {len(available_doc_ids)} 文档")

    cases = load_cases()
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"待处理题目: {len(cases)} 题 (split={split})")

    # 并发处理
    results: list[dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_index = {
            pool.submit(process_one_case, case, available_doc_ids, chunks): idx
            for idx, case in enumerate(cases)
        }
        done_count = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                case = cases[idx]
                results[idx] = {
                    "qid": case["qid"], "domain": case["domain"], "answer": "A",
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "duration_ms": 0, "evidence": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            done_count += 1
            r = results[idx]
            sc = ""
            print(f"  [{done_count}/{len(cases)}] {r['qid']} answer={normalize_answer(r['answer'])}{sc} "
                  f"tokens={r['total_tokens']} error={r['error'] or 'none'}", flush=True)

    elapsed = time.perf_counter() - t0

    # 生成提交文件
    submission_path = output_dir / f"submission_{split}.csv"
    results_path = output_dir / f"results_{split}.jsonl"
    evidence_path = output_dir / f"evidence_{split}.json"

    write_submission_csv(submission_path, results)
    write_results_jsonl(results_path, results)
    write_evidence_json(evidence_path, results)

    # 汇总统计
    total_tokens = sum(r["total_tokens"] for r in results)
    llm_success = sum(1 for r in results if not r["error"])
    avg_duration = sum(r["duration_ms"] for r in results) / len(results)

    print()
    print("=" * 55)
    print(f"完成: {len(results)} 题, 耗时 {elapsed:.0f}s ({elapsed / 60:.1f}分钟)")
    print(f"LLM 成功: {llm_success}/{len(results)}")
    print(f"总 Token: {total_tokens:,} (预算 5,000,000, 使用 {total_tokens * 100 / 5_000_000:.1f}%)")
    print(f"平均每题: {total_tokens // len(results):,} tokens, {avg_duration:.0f}ms")
    print(f"提交文件: {submission_path}")
    print(f"  - submission_{split}.csv (含 summary 行)")
    print(f"  - results_{split}.jsonl")
    print(f"  - evidence_{split}.json")
    print("=" * 55)


if __name__ == "__main__":
    main()
