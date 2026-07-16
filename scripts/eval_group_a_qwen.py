from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.formatter import build_evidence_items, choose_answer, needs_self_check
from api.services.qwen_client import answer_with_qwen, get_qwen_config_status, QwenAnswer, verify_with_qwen
from api.services.retrieval_loop import run_retrieval_loop

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "group_a_qwen_eval"
ANSWER_CSV = OUTPUT_DIR / "answer.csv"

ANSWER_FORMAT_MAPPING = {
    "single": "single",
    "mcq": "single",
    "multi": "multi",
    "tf": "judge",
    "judge": "judge",
}
OPTION_KEYS = ("A", "B", "C", "D", "E", "F")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_options(raw_options: Any) -> list[str] | None:
    if isinstance(raw_options, dict):
        ordered = [str(raw_options[key]) for key in OPTION_KEYS if key in raw_options]
        return ordered or None
    if isinstance(raw_options, list):
        return [str(item) for item in raw_options] or None
    return None


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for file_path in sorted(QUESTIONS_DIR.glob("*.json")):
        payload = read_json(file_path)
        for raw_case in payload:
            options = normalize_options(raw_case.get("options"))
            answer_format = ANSWER_FORMAT_MAPPING.get(str(raw_case.get("answer_format", "")).lower())
            if not options or not answer_format:
                continue
            cases.append(
                {
                    "file": file_path.name,
                    "qid": str(raw_case["qid"]),
                    "domain": str(raw_case.get("domain", "unknown")),
                    "question": str(raw_case["question"]),
                    "options": options,
                    "answer_format": answer_format,
                    "doc_ids": [str(item) for item in raw_case.get("doc_ids", [])],
                    "scenario_type": str(raw_case.get("type", "unknown")),
                }
            )
    return cases


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(int(len(ordered) * ratio) - 1, 0)
    return float(ordered[index])


def build_output_paths(suffix: str) -> tuple[Path, Path, Path]:
    normalized = f"_{suffix}" if suffix else ""
    return (
        OUTPUT_DIR / f"results{normalized}.jsonl",
        OUTPUT_DIR / f"summary{normalized}.json",
        OUTPUT_DIR / f"report{normalized}.md",
    )


def render_report(summary: dict[str, Any]) -> str:
    token_usage = summary["token_usage"]
    latency = summary["latency"]
    proxy_metrics = summary["proxy_metrics"]
    lines = [
        "# Group A Qwen 全量评测",
        "",
        f"- 样本目录：`{summary['sample']}`",
        f"- 样本数：`{summary['case_count']}`",
        "- 真实准确率：`不可计算`",
        f"- 原因：{summary['true_accuracy_note']}",
        f"- 规则基线一致率：`{proxy_metrics['fallback_consistency_rate']}` ({proxy_metrics['fallback_consistency_count']}/{summary['case_count']})",
        f"- LLM 成功率：`{proxy_metrics['llm_success_rate']}` ({proxy_metrics['llm_success_count']}/{summary['case_count']})",
        "",
        "## Token 统计",
        f"- Prompt Tokens：`{token_usage['total_prompt_tokens']}`",
        f"- Completion Tokens：`{token_usage['total_completion_tokens']}`",
        f"- Total Tokens：`{token_usage['total_tokens']}`",
        f"- 平均每题 Prompt Tokens：`{token_usage['avg_prompt_tokens']}`",
        f"- 平均每题 Completion Tokens：`{token_usage['avg_completion_tokens']}`",
        f"- 平均每题 Total Tokens：`{token_usage['avg_total_tokens']}`",
        f"- P50 Total Tokens：`{token_usage['p50_total_tokens']}`",
        f"- P95 Total Tokens：`{token_usage['p95_total_tokens']}`",
        "",
        "## 时延统计",
        f"- 平均耗时：`{latency['avg_duration_ms']} ms`",
        f"- P50 耗时：`{latency['p50_duration_ms']} ms`",
        f"- P95 耗时：`{latency['p95_duration_ms']} ms`",
        "",
        "## 按 Domain",
    ]
    for domain, aggregate in summary["by_domain"].items():
        lines.append(
            f"- `{domain}`：cases=`{aggregate['cases']}`，avg_total_tokens=`{aggregate['avg_total_tokens']}`，fallback_consistency_rate=`{aggregate['fallback_consistency_rate']}`，avg_duration_ms=`{aggregate['avg_duration_ms']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--chunks-file", type=str, default="chunks_merged.jsonl",
                        help="chunks file to use (chunks_merged.jsonl or chunks.jsonl)")
    args = parser.parse_args()
    qwen_status = get_qwen_config_status()
    if not qwen_status.enabled:
        missing = ",".join(qwen_status.missing_settings) or "none"
        raise RuntimeError(f"Qwen 未启用，missing_settings={missing} disable_reason={qwen_status.disable_reason}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_jsonl, summary_json, summary_md = build_output_paths(args.suffix)
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / args.chunks_file)]
    print(f"Using chunks: {args.chunks_file} ({len(chunks)} chunks)")
    all_cases = load_cases()
    if args.limit > 0:
        cases = all_cases[args.offset : args.offset + args.limit]
    else:
        cases = all_cases[args.offset :]
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        request = RunQuestionTaskRequest(
            mode="A",
            qid=case["qid"],
            question=case["question"],
            options=case["options"],
            answerFormat=case["answer_format"],
            docIds=case["doc_ids"],
        )
        retrieval = run_retrieval_loop(
            request,
            available_doc_ids=available_doc_ids,
            structured_chunks=chunks,
        )
        fallback_answer = choose_answer(request.answer_format, retrieval["reasoning_results"])
        evidence_items = build_evidence_items(retrieval["reasoning_results"], retrieval["evidence_map"])

        qwen_error = None
        qwen_result = None
        try:
            qwen_result = answer_with_qwen(
                question=request.question,
                options=request.options,
                answer_format=request.answer_format,
                evidence=list(retrieval["candidate_chunks"]),
                reasoning_hints=list(retrieval["reasoning_results"]),
            )
        except Exception as exc:  # noqa: BLE001
            qwen_error = f"{type(exc).__name__}: {exc}"

        final_answer = qwen_result.answer if qwen_result is not None else fallback_answer

        # Self-check for low-confidence answers
        self_checked = False
        verify_error = None
        if qwen_result is not None and needs_self_check(
            request.answer_format, list(retrieval["reasoning_results"]), final_answer,
        ):
            self_checked = True
            try:
                verified = verify_with_qwen(
                    question=request.question,
                    options=request.options,
                    answer_format=request.answer_format,
                    evidence=list(retrieval["candidate_chunks"]),
                    reasoning_hints=list(retrieval["reasoning_results"]),
                    first_answer=final_answer,
                )
                if verified is not None:
                    final_answer = verified.answer
                    # Add verification tokens to total
                    qwen_result = QwenAnswer(
                        answer=final_answer,
                        reasoning=verified.reasoning,
                        prompt_tokens=qwen_result.prompt_tokens + verified.prompt_tokens,
                        completion_tokens=qwen_result.completion_tokens + verified.completion_tokens,
                        total_tokens=qwen_result.total_tokens + verified.total_tokens,
                        model=verified.model,
                    )
            except Exception as exc:  # noqa: BLE001
                verify_error = f"{type(exc).__name__}: {exc}"

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        row = {
            "qid": case["qid"],
            "domain": case["domain"],
            "scenario_type": case["scenario_type"],
            "file": case["file"],
            "answer": final_answer,
            "fallback_answer": fallback_answer,
            "matches_fallback": final_answer == fallback_answer,
            "prompt_tokens": int(qwen_result.prompt_tokens) if qwen_result is not None else 0,
            "completion_tokens": int(qwen_result.completion_tokens) if qwen_result is not None else 0,
            "total_tokens": int(qwen_result.total_tokens) if qwen_result is not None else 0,
            "duration_ms": duration_ms,
            "candidate_doc_count": len(retrieval["doc_ids"]),
            "candidate_chunk_count": len(retrieval["candidate_chunks"]),
            "evidence_count": len(evidence_items),
            "llm_model": qwen_result.model if qwen_result is not None else "none",
            "llm_error": qwen_error,
            "self_checked": self_checked,
            "verify_error": verify_error,
            "retrieval_quality": retrieval["evaluation"].get("quality_score", 0.0),
            "retrieval_quality_label": retrieval["evaluation"].get("quality_label", "unknown"),
        }
        results.append(row)
        sc_tag = " [self-check]" if self_checked else ""
        print(
            f"[{index}/{len(cases)}] {case['qid']} answer={final_answer}{sc_tag} "
            f"tokens={row['total_tokens']} error={qwen_error or 'none'}"
        )

    results_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")

    # 生成 answer.csv（赛题提交格式：qid, answer, prompt_tokens, completion_tokens, total_tokens）
    with ANSWER_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        for row in results:
            answer = row["answer"]
            # 清洗答案：提取大写字母，排序去重，禁止 N/A 或空值
            letters = "".join(sorted(set(re.findall(r"[A-F]", (answer or "").upper()))))
            if not letters:
                letters = "A"  # 兜底：避免空答案被判非法
            writer.writerow([row["qid"], letters, row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]])
    print(f"answer.csv 已生成: {ANSWER_CSV}")

    token_rows = [row for row in results if row["total_tokens"] > 0]
    durations = [row["duration_ms"] for row in results]
    consistency_count = sum(1 for row in results if row["matches_fallback"])

    by_domain: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "llm_success": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "fallback_consistency": 0,
            "avg_duration_ms": 0.0,
            "avg_total_tokens": 0.0,
            "fallback_consistency_rate": 0.0,
        }
    )
    domain_durations: dict[str, list[float]] = defaultdict(list)
    for row in results:
        aggregate = by_domain[row["domain"]]
        aggregate["cases"] += 1
        aggregate["llm_success"] += 1 if row["total_tokens"] > 0 else 0
        aggregate["prompt_tokens"] += row["prompt_tokens"]
        aggregate["completion_tokens"] += row["completion_tokens"]
        aggregate["total_tokens"] += row["total_tokens"]
        aggregate["fallback_consistency"] += 1 if row["matches_fallback"] else 0
        domain_durations[row["domain"]].append(row["duration_ms"])

    for domain, aggregate in by_domain.items():
        aggregate["avg_duration_ms"] = round(sum(domain_durations[domain]) / len(domain_durations[domain]), 2)
        aggregate["avg_total_tokens"] = round(aggregate["total_tokens"] / aggregate["cases"], 2) if aggregate["cases"] else 0.0
        aggregate["fallback_consistency_rate"] = (
            round(aggregate["fallback_consistency"] / aggregate["cases"], 4) if aggregate["cases"] else 0.0
        )

    summary = {
        "sample": "public_dataset_a/questions/group_a",
        "chunks_file": args.chunks_file,
        "chunks_count": len(chunks),
        "offset": args.offset,
        "limit": args.limit,
        "case_count": len(results),
        "true_accuracy_available": False,
        "true_accuracy_note": "题库未提供 gold_answer/labels，工作区内未发现可覆盖 100 题的标准答案，因此无法计算真实准确率。",
        "proxy_metrics": {
            "fallback_consistency_count": consistency_count,
            "fallback_consistency_rate": round(consistency_count / len(results), 4) if results else 0.0,
            "llm_success_count": len(token_rows),
            "llm_success_rate": round(len(token_rows) / len(results), 4) if results else 0.0,
        },
        "token_usage": {
            "total_prompt_tokens": sum(row["prompt_tokens"] for row in token_rows),
            "total_completion_tokens": sum(row["completion_tokens"] for row in token_rows),
            "total_tokens": sum(row["total_tokens"] for row in token_rows),
            "avg_prompt_tokens": round(sum(row["prompt_tokens"] for row in token_rows) / len(token_rows), 2)
            if token_rows
            else 0.0,
            "avg_completion_tokens": round(sum(row["completion_tokens"] for row in token_rows) / len(token_rows), 2)
            if token_rows
            else 0.0,
            "avg_total_tokens": round(sum(row["total_tokens"] for row in token_rows) / len(token_rows), 2)
            if token_rows
            else 0.0,
            "p50_total_tokens": round(statistics.median([row["total_tokens"] for row in token_rows]), 2)
            if token_rows
            else 0.0,
            "p95_total_tokens": round(percentile([row["total_tokens"] for row in token_rows], 0.95), 2)
            if token_rows
            else 0.0,
        },
        "latency": {
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "p50_duration_ms": round(statistics.median(durations), 2) if durations else 0.0,
            "p95_duration_ms": round(percentile(durations, 0.95), 2) if durations else 0.0,
        },
        "by_domain": dict(sorted(by_domain.items())),
        "output_files": {
            "results_jsonl": str(results_jsonl),
            "summary_json": str(summary_json),
            "report_md": str(summary_md),
            "answer_csv": str(ANSWER_CSV),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
