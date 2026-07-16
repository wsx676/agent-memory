from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.formatter import build_evidence_items, choose_answer
from api.services.planner import build_plan
from api.services.qwen_client import answer_with_qwen, get_qwen_config_status
from api.services.reasoner import reason_options
from api.services.retriever import rank_chunks


QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
CHUNKS_FILE = PREPROCESSED_DIR / "chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "qwen_smoke_test"
RESULTS_JSONL = OUTPUT_DIR / "results.jsonl"
REPORT_MD = OUTPUT_DIR / "report.md"
ANSWER_FORMAT_MAPPING = {"single": "single", "mcq": "single", "multi": "multi", "tf": "judge", "judge": "judge"}
OPTION_KEYS = ("A", "B", "C", "D", "E", "F")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_candidate_chunks(doc_ids: list[str]) -> list[StructuredChunk]:
    target_doc_ids = set(doc_ids)
    if not target_doc_ids:
        return []
    chunks: list[StructuredChunk] = []
    with CHUNKS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("docId")) in target_doc_ids:
                chunks.append(StructuredChunk.model_validate(row))
    return chunks


def _normalize_options(raw_options: Any) -> list[str] | None:
    if isinstance(raw_options, dict):
        ordered = [str(raw_options[key]) for key in OPTION_KEYS if key in raw_options]
        return ordered or None
    if isinstance(raw_options, list):
        return [str(item) for item in raw_options] or None
    return None


def _collect_cases(offset: int, limit: int, qid: str | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen = 0
    for file_path in sorted(QUESTIONS_DIR.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for raw_case in payload:
            current_qid = str(raw_case.get("qid", ""))
            if qid and current_qid != qid:
                continue
            if seen < offset:
                seen += 1
                continue
            if len(cases) >= limit:
                return cases
            options = _normalize_options(raw_case.get("options"))
            answer_format = ANSWER_FORMAT_MAPPING.get(str(raw_case.get("answer_format", "")).lower())
            if not options or not answer_format:
                seen += 1
                continue
            cases.append(
                {
                    "qid": current_qid,
                    "domain": str(raw_case.get("domain", "unknown")),
                    "question": str(raw_case["question"]),
                    "options": options,
                    "answerFormat": answer_format,
                    "docIds": [str(item) for item in raw_case.get("doc_ids", [])],
                    "mode": "A",
                }
            )
            seen += 1
    return cases


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen Smoke Test Report",
        "",
        f"- 测试题数：`{len(rows)}`",
        f"- 结果文件：`{RESULTS_JSONL}`",
        "",
        "## 结果概览",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['qid']}",
                f"- Domain：`{row['domain']}`",
                f"- Answer：`{row['answer']}`",
                f"- Evidence Count：`{row['evidence_count']}`",
                f"- Prompt Tokens：`{row['prompt_tokens']}`",
                f"- Completion Tokens：`{row['completion_tokens']}`",
                f"- Total Tokens：`{row['total_tokens']}`",
                f"- LLM Used：`{row['llm_used']}`",
                f"- LLM Model：`{row['llm_model']}`",
                f"- Logs：`{' | '.join(row['logs'])}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="最多运行多少道题")
    parser.add_argument("--offset", type=int, default=0, help="从第几题开始采样")
    parser.add_argument("--append", action="store_true", help="追加到已有结果文件")
    parser.add_argument("--qid", type=str, default="", help="只运行指定 qid")
    args = parser.parse_args()

    qwen_status = get_qwen_config_status()
    if not qwen_status.enabled:
        missing = ",".join(qwen_status.missing_settings) or "none"
        raise RuntimeError(f"Qwen 未启用，missing_settings={missing} disable_reason={qwen_status.disable_reason}")

    rows: list[dict[str, Any]] = _read_jsonl(RESULTS_JSONL) if args.append and RESULTS_JSONL.exists() else []
    new_rows: list[dict[str, Any]] = []
    for case in _collect_cases(args.offset, args.limit, args.qid or None):
        request = RunQuestionTaskRequest.model_validate(case)
        plan = build_plan(request)
        plan["domain"] = case["domain"]
        chunks = _load_candidate_chunks(request.doc_ids or [])
        candidate_chunks = rank_chunks(request.question, request.options, chunks, request.doc_ids or None, plan)
        reasoning_results, evidence_map = reason_options(request.options, candidate_chunks)
        fallback_answer = choose_answer(request.answer_format, reasoning_results)
        qwen_result = None
        qwen_error = None
        try:
            qwen_result = answer_with_qwen(
                question=request.question,
                options=request.options,
                answer_format=request.answer_format,
                evidence=candidate_chunks,
                reasoning_hints=reasoning_results,
            )
        except Exception as exc:  # noqa: BLE001
            qwen_error = f"{type(exc).__name__}: {exc}"
        evidence_items = build_evidence_items(reasoning_results, evidence_map)
        logs = [
            f"candidate_doc_count={len(request.doc_ids or [])}",
            f"candidate_chunk_count={len(candidate_chunks)}",
            f"fallback_answer={fallback_answer}",
            f"llm_used={'qwen' if qwen_result is not None else 'local_rule'}",
            f"llm_model={qwen_result.model if qwen_result is not None else 'none'}",
            f"llm_error={qwen_error or 'none'}",
        ]
        row = {
            "qid": request.qid,
            "domain": case["domain"],
            "answer": qwen_result.answer if qwen_result is not None else fallback_answer,
            "evidence_count": len(evidence_items),
            "prompt_tokens": qwen_result.prompt_tokens if qwen_result is not None else 0,
            "completion_tokens": qwen_result.completion_tokens if qwen_result is not None else 0,
            "total_tokens": qwen_result.total_tokens if qwen_result is not None else 0,
            "llm_used": "qwen" if qwen_result is not None else "local_rule",
            "llm_model": qwen_result.model if qwen_result is not None else "none",
            "logs": logs,
        }
        rows.append(row)
        new_rows.append(row)
        print(
            f"[done] qid={request.qid} answer={row['answer']} "
            f"fallback={fallback_answer} chunks={len(candidate_chunks)} model={row['llm_model']}"
        )
        _write_jsonl(RESULTS_JSONL, rows)
        _write_text(REPORT_MD, _render_report(rows))

    print(f"[saved] added={len(new_rows)} total={len(rows)} results={RESULTS_JSONL}")


if __name__ == "__main__":
    main()
