from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.formatter import build_evidence_items, choose_answer
from api.services.planner import build_plan
from api.services.reasoner import reason_options
from api.services.retriever import rank_chunks


QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
DOCUMENTS_FILE = PREPROCESSED_DIR / "documents.jsonl"
CHUNKS_FILE = PREPROCESSED_DIR / "chunks.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "group_a"
INVENTORY_FILE = OUTPUT_DIR / "group_a_case_inventory.json"
LOG_FILE = OUTPUT_DIR / "group_a_execution_log.jsonl"
REPORT_FILE = OUTPUT_DIR / "group_a_test_report.md"

REQUIRED_FIELDS = {"qid", "domain", "split", "question", "options", "answer_format", "type", "doc_ids"}
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
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第 {line_number} 行 JSON 解析失败: {exc}") from exc
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_options(raw_options: Any) -> tuple[list[str] | None, list[str]]:
    issues: list[str] = []
    if isinstance(raw_options, dict):
        ordered = [raw_options[key] for key in OPTION_KEYS if key in raw_options]
        if not ordered:
            issues.append("options_dict_empty")
            return None, issues
        if set(raw_options) - set(OPTION_KEYS):
            issues.append("options_dict_has_nonstandard_keys")
        return ordered, issues
    if isinstance(raw_options, list):
        if not raw_options:
            issues.append("options_list_empty")
            return None, issues
        return [str(item) for item in raw_options], issues
    issues.append("options_not_list_or_dict")
    return None, issues


def normalize_answer_format(raw_answer_format: Any) -> tuple[str | None, list[str]]:
    issues: list[str] = []
    if not isinstance(raw_answer_format, str):
        issues.append("answer_format_not_string")
        return None, issues
    normalized = ANSWER_FORMAT_MAPPING.get(raw_answer_format.lower())
    if normalized is None:
        issues.append("answer_format_unsupported")
    elif normalized != raw_answer_format:
        issues.append("answer_format_requires_mapping")
    return normalized, issues


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(status, 9)


def load_preprocessed_assets() -> tuple[set[str], set[str], list[StructuredChunk]]:
    documents = read_jsonl(DOCUMENTS_FILE)
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks = [StructuredChunk.model_validate(item) for item in read_jsonl(CHUNKS_FILE)]
    chunk_doc_ids = {chunk.doc_id for chunk in chunks}
    return available_doc_ids, chunk_doc_ids, chunks


def collect_question_files() -> list[Path]:
    return sorted(QUESTIONS_DIR.glob("*.json"))


def validate_and_collect_cases(
    question_files: list[Path],
    available_doc_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    file_validations: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    execution_logs: list[dict[str, Any]] = []
    issue_counter: Counter[str] = Counter()
    seen_qids: set[str] = set()

    for file_path in question_files:
        file_issues: list[str] = []
        try:
            payload = read_json(file_path)
        except Exception as exc:  # noqa: BLE001
            execution_logs.append(
                {
                    "file": file_path.name,
                    "phase": "validation",
                    "status": "FAIL",
                    "issues": ["file_unreadable"],
                    "error": repr(exc),
                }
            )
            file_validations.append(
                {
                    "file": file_path.name,
                    "status": "FAIL",
                    "question_count": 0,
                    "issues": ["file_unreadable"],
                }
            )
            issue_counter["file_unreadable"] += 1
            continue

        if not isinstance(payload, list):
            file_issues.append("root_not_list")
            payload = []

        for raw_case in payload:
            case_issues: list[str] = []
            if not isinstance(raw_case, dict):
                case_issues.append("case_not_object")
                issue_counter.update(case_issues)
                execution_logs.append(
                    {
                        "file": file_path.name,
                        "phase": "validation",
                        "status": "FAIL",
                        "issues": case_issues,
                    }
                )
                continue

            missing_fields = sorted(REQUIRED_FIELDS - set(raw_case))
            if missing_fields:
                case_issues.extend(f"missing_field:{field}" for field in missing_fields)

            qid = str(raw_case.get("qid", f"{file_path.stem}-unknown"))
            if qid in seen_qids:
                case_issues.append("duplicate_qid")
            else:
                seen_qids.add(qid)

            options, option_issues = normalize_options(raw_case.get("options"))
            case_issues.extend(option_issues)

            normalized_answer_format, format_issues = normalize_answer_format(raw_case.get("answer_format"))
            case_issues.extend(format_issues)

            doc_ids = raw_case.get("doc_ids")
            if not isinstance(doc_ids, list) or not doc_ids:
                case_issues.append("doc_ids_invalid")
                doc_ids = []

            missing_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in available_doc_ids]
            if missing_doc_ids:
                case_issues.append("missing_doc_reference")

            if options is not None and not 2 <= len(options) <= 6:
                case_issues.append("options_count_out_of_range")

            case_record = {
                "file": file_path.name,
                "qid": qid,
                "domain": raw_case.get("domain"),
                "split": raw_case.get("split"),
                "scenario_type": raw_case.get("type"),
                "raw_answer_format": raw_case.get("answer_format"),
                "normalized_answer_format": normalized_answer_format,
                "doc_ids": doc_ids,
                "missing_doc_ids": missing_doc_ids,
                "doc_count": len(doc_ids),
                "question_length": len(str(raw_case.get("question", ""))),
                "option_count": len(options or []),
                "issues": case_issues,
            }
            cases.append(
                {
                    **case_record,
                    "question": str(raw_case.get("question", "")),
                    "options": options,
                }
            )
            if case_issues:
                execution_logs.append(
                    {
                        "file": file_path.name,
                        "qid": qid,
                        "phase": "validation",
                        "status": "WARN",
                        "issues": case_issues,
                        "missing_doc_ids": missing_doc_ids,
                    }
                )
            issue_counter.update(case_issues)

        file_status = "PASS" if not file_issues else "WARN"
        file_validations.append(
            {
                "file": file_path.name,
                "status": file_status,
                "question_count": len(payload),
                "issues": file_issues,
            }
        )
        issue_counter.update(file_issues)

    return file_validations, cases, execution_logs, issue_counter


def build_case_request(case: dict[str, Any]) -> RunQuestionTaskRequest:
    return RunQuestionTaskRequest(
        mode="A",
        qid=case["qid"],
        question=case["question"],
        options=case["options"],
        answerFormat=case["normalized_answer_format"],
        docIds=case["doc_ids"],
    )


def execute_case(case: dict[str, Any], available_doc_ids: set[str], chunks: list[StructuredChunk]) -> dict[str, Any]:
    start = time.perf_counter()
    if not case["normalized_answer_format"] or not case["options"]:
        return {
            "qid": case["qid"],
            "file": case["file"],
            "domain": case["domain"],
            "scenario_type": case["scenario_type"],
            "status": "FAIL",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "answer": None,
            "evidence_count": 0,
            "candidate_chunk_count": 0,
            "missing_doc_ids": case["missing_doc_ids"],
            "issues": case["issues"],
            "error": "question_format_invalid",
        }

    request = build_case_request(case)
    try:
        plan = build_plan(request)
        plan["domain"] = str(case["domain"] or plan["domain"])
        candidate_doc_ids = [doc_id for doc_id in case["doc_ids"] if doc_id in available_doc_ids]
        candidate_chunks = rank_chunks(
            request.question,
            request.options,
            chunks,
            candidate_doc_ids or None,
            plan,
        )
        reasoning_results, evidence_map = reason_options(request.options, candidate_chunks)
        answer = choose_answer(request.answer_format, reasoning_results)
        evidence_items = build_evidence_items(reasoning_results, evidence_map)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        issues = list(case["issues"])
        if not candidate_doc_ids:
            issues.append("candidate_doc_ids_empty")
        if not candidate_chunks:
            issues.append("candidate_chunks_empty")
        if not evidence_items:
            issues.append("evidence_empty")

        status = "PASS"
        if any(issue.startswith("missing_field:") for issue in issues):
            status = "FAIL"
        elif any(issue in {"candidate_doc_ids_empty", "candidate_chunks_empty", "evidence_empty"} for issue in issues):
            status = "WARN"
        elif case["missing_doc_ids"]:
            status = "WARN"
        elif case["issues"]:
            status = "WARN"

        return {
            "qid": case["qid"],
            "file": case["file"],
            "domain": case["domain"],
            "scenario_type": case["scenario_type"],
            "status": status,
            "duration_ms": duration_ms,
            "answer": answer,
            "evidence_count": len(evidence_items),
            "candidate_chunk_count": len(candidate_chunks),
            "candidate_doc_ids": candidate_doc_ids,
            "missing_doc_ids": case["missing_doc_ids"],
            "issues": issues,
            "top_evidence_preview": [item.quoted_text[:80] for item in evidence_items[:2]],
            "logs": [
                f"raw_answer_format={case['raw_answer_format']}",
                f"normalized_answer_format={case['normalized_answer_format']}",
                f"candidate_doc_count={len(candidate_doc_ids)}",
                f"candidate_chunk_count={len(candidate_chunks)}",
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "qid": case["qid"],
            "file": case["file"],
            "domain": case["domain"],
            "scenario_type": case["scenario_type"],
            "status": "FAIL",
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "answer": None,
            "evidence_count": 0,
            "candidate_chunk_count": 0,
            "missing_doc_ids": case["missing_doc_ids"],
            "issues": case["issues"],
            "error": repr(exc),
        }


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(row.get(key, "unknown"))][row["status"]] += 1
    summary: list[dict[str, Any]] = []
    for name, counts in sorted(grouped.items()):
        total = sum(counts.values())
        summary.append(
            {
                key: name,
                "total": total,
                "pass": counts.get("PASS", 0),
                "warn": counts.get("WARN", 0),
                "fail": counts.get("FAIL", 0),
                "pass_rate": round(counts.get("PASS", 0) / total, 4) if total else 0.0,
            }
        )
    return summary


def render_markdown_report(
    file_validations: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    execution_logs: list[dict[str, Any]],
    issue_counter: Counter[str],
    corpus_summary: dict[str, Any],
) -> str:
    execution_rows = [row for row in execution_logs if row.get("phase") != "validation"]
    pass_count = sum(1 for row in execution_rows if row["status"] == "PASS")
    warn_count = sum(1 for row in execution_rows if row["status"] == "WARN")
    fail_count = sum(1 for row in execution_rows if row["status"] == "FAIL")
    total_cases = len(execution_rows)
    durations = [row["duration_ms"] for row in execution_rows]
    avg_duration = round(sum(durations) / len(durations), 2) if durations else 0.0
    p95_index = max(int(len(durations) * 0.95) - 1, 0) if durations else 0
    p95_duration = round(sorted(durations)[p95_index], 2) if durations else 0.0
    domain_summary = summarize_by_key(execution_rows, "domain")
    type_summary = summarize_by_key(execution_rows, "scenario_type")
    top_issues = issue_counter.most_common(12)
    warnings = sorted(
        [row for row in execution_rows if row["status"] != "PASS"],
        key=lambda row: (status_rank(row["status"]), -row.get("duration_ms", 0), row.get("qid", "")),
    )[:20]

    lines = [
        "# Group A 问题文件测试报告",
        "",
        "## 测试范围",
        f"- 测试目录：`{QUESTIONS_DIR}`",
        f"- 问题文件数：`{len(file_validations)}`",
        f"- 问题总数：`{len(cases)}`",
        f"- 覆盖 domain：`{', '.join(sorted({str(case['domain']) for case in cases}))}`",
        f"- 覆盖场景类型数：`{len({str(case['scenario_type']) for case in cases})}`",
        f"- 预处理文档数：`{corpus_summary['document_count']}`",
        f"- chunk 覆盖文档数：`{corpus_summary['chunk_document_count']}`",
        f"- 文档-分块断链数：`{corpus_summary['documents_without_chunks']}`",
        "",
        "## 文件有效性校验",
        f"- 可读取文件：`{sum(1 for item in file_validations if item['status'] == 'PASS')}/{len(file_validations)}`",
        f"- 文件级告警：`{sum(1 for item in file_validations if item['status'] == 'WARN')}`",
        f"- 文件级失败：`{sum(1 for item in file_validations if item['status'] == 'FAIL')}`",
    ]
    for item in file_validations:
        issue_text = ",".join(item["issues"]) if item["issues"] else "none"
        lines.append(f"- `{item['file']}`：status=`{item['status']}`，questions=`{item['question_count']}`，issues=`{issue_text}`")

    lines.extend(
        [
            "",
            "## 用例执行结果",
            f"- 已执行用例：`{total_cases}`",
            f"- 通过：`{pass_count}`",
            f"- 告警：`{warn_count}`",
            f"- 失败：`{fail_count}`",
            f"- 执行通过率：`{round(pass_count / total_cases * 100, 2) if total_cases else 0.0}%`",
            f"- 平均耗时：`{avg_duration} ms`",
            f"- P95 耗时：`{p95_duration} ms`",
            "",
            "## 场景覆盖统计",
            "### 按 Domain",
        ]
    )
    for row in domain_summary:
        lines.append(
            f"- `{row['domain']}`：total=`{row['total']}`，pass=`{row['pass']}`，warn=`{row['warn']}`，fail=`{row['fail']}`，pass_rate=`{row['pass_rate']}`"
        )

    lines.append("")
    lines.append("### 按场景类型")
    for row in type_summary:
        lines.append(
            f"- `{row['scenario_type']}`：total=`{row['total']}`，pass=`{row['pass']}`，warn=`{row['warn']}`，fail=`{row['fail']}`，pass_rate=`{row['pass_rate']}`"
        )

    lines.extend(["", "## 异常与适配问题统计"])
    if top_issues:
        for issue, count in top_issues:
            lines.append(f"- `{issue}`：`{count}`")
    else:
        lines.append("- 无异常问题。")

    lines.extend(["", "## 重点异常样本"])
    if warnings:
        for row in warnings:
            lines.append(
                f"- `{row['qid']}`：status=`{row['status']}`，issues=`{','.join(row.get('issues', [])) or 'none'}`，missing_doc_ids=`{','.join(row.get('missing_doc_ids', [])) or 'none'}`"
            )
    else:
        lines.append("- 无告警或失败样本。")

    lines.extend(
        [
            "",
            "## 适配优化建议",
            "- 统一题库 `answer_format` 枚举，建议只保留 `single/multi/judge`，避免当前 `mcq/tf/multi` 混用带来的适配分支。",
            "- 统一 `options` 结构，建议固定为数组或固定键序的对象，并在数据契约中明确选项顺序规则。",
            "- 为题库补充标准答案字段，如 `gold_answer` 或 `labels`，否则目前只能评估执行通过率，无法计算准确率与召回率。",
            "- 统一 `type` 字段词表，目前中英文、自由文本并存，建议建立可控枚举并维护场景映射表。",
            "- 对 `doc_ids` 建立发布前校验，确保题库引用的文档标识与预处理产物中的 `doc_id` 完全一致。",
            "- 对预处理产物增加 `documents.jsonl` 与 `chunks.jsonl` 的一致性校验，当前存在文档已登记但 chunk 未落盘的断链问题。",
            "- 对保险计算题、财报数值题补充结构化数值标签，后续便于精确评估数值抽取和计算链路。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    available_doc_ids, chunk_doc_ids, chunks = load_preprocessed_assets()
    question_files = collect_question_files()
    file_validations, cases, validation_logs, issue_counter = validate_and_collect_cases(question_files, available_doc_ids)

    execution_rows = []
    for case in cases:
        result = execute_case(case, available_doc_ids, chunks)
        execution_rows.append(result)
        issue_counter.update(result.get("issues", []))
        if result.get("error"):
            issue_counter["execution_exception"] += 1

    inventory = {
        "question_dir": str(QUESTIONS_DIR),
        "question_files": [path.name for path in question_files],
        "corpus_summary": {
            "document_count": len(available_doc_ids),
            "chunk_document_count": len(chunk_doc_ids),
            "documents_without_chunks": len(available_doc_ids - chunk_doc_ids),
            "sample_documents_without_chunks": sorted(available_doc_ids - chunk_doc_ids)[:50],
        },
        "file_validations": file_validations,
        "case_count": len(cases),
        "cases": cases,
    }
    report = render_markdown_report(
        file_validations,
        cases,
        validation_logs + execution_rows,
        issue_counter,
        inventory["corpus_summary"],
    )

    write_json(INVENTORY_FILE, inventory)
    write_jsonl(LOG_FILE, validation_logs + execution_rows)
    write_text(REPORT_FILE, report)

    print(json.dumps({"report": str(REPORT_FILE), "log": str(LOG_FILE), "inventory": str(INVENTORY_FILE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
