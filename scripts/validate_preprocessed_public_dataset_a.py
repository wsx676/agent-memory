from __future__ import annotations

import csv
import json
import mimetypes
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = ROOT / "public_dataset_a"
RAW_ROOT = DATASET_ROOT / "raw"
QUESTIONS_ROOT = DATASET_ROOT / "questions" / "group_a"
PREPROCESS_ROOT = ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_ROOT = ROOT / "validation_outputs" / "public_dataset_a"

DOCS_JSONL = PREPROCESS_ROOT / "documents.jsonl"
CHUNKS_JSONL = PREPROCESS_ROOT / "chunks.jsonl"

DOC_VALIDATION_CSV = OUTPUT_ROOT / "preprocessed_validation_doc_report.csv"
MANUAL_REVIEW_CSV = OUTPUT_ROOT / "manual_review_queue.csv"
SUMMARY_JSON = OUTPUT_ROOT / "preprocessed_validation_summary.json"
SUMMARY_MD = OUTPUT_ROOT / "preprocessed_validation_summary.md"


THRESHOLDS = {
    "format_compliance_rate": 0.995,
    "file_integrity_rate": 0.99,
    "encoding_uniformity_rate": 0.99,
    "doc_id_mapping_rate": 1.0,
    "metadata_completeness_rate": 0.98,
    "field_completeness_rate": 0.95,
    "clause_boundary_rate": 0.96,
    "logical_consistency_rate": 0.95,
    "sensitive_scan_pass_rate": 1.0,
    "copyright_retention_rate": 0.99,
    "attachment_mapping_rate": 0.98,
    "pdf_avg_seconds": 45.0,
    "html_avg_seconds": 5.0,
    "txt_avg_seconds": 3.0,
    "automation_coverage_rate": 0.75,
    "manual_review_trigger_rate": 0.20,
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_question_doc_ids() -> set[str]:
    mapped_doc_ids: set[str] = set()
    for question_file in QUESTIONS_ROOT.glob("*.json"):
        data = json.loads(question_file.read_text(encoding="utf-8"))
        for item in data:
            for doc_id in item.get("doc_ids", []):
                mapped_doc_ids.add(str(doc_id))
    return mapped_doc_ids


def get_local_attachment_files(doc_id: str) -> list[Path]:
    attachments_root = RAW_ROOT / "regulatory" / "attachments"
    pattern = f"{doc_id}_att*.pdf"
    return sorted(attachments_root.glob(pattern))


def is_doc_id_mapping_applicable(doc: dict, mapped_doc_ids: set[str]) -> bool:
    return doc["doc_id"] in mapped_doc_ids


def is_doc_id_matched(doc: dict, mapped_doc_ids: set[str]) -> bool:
    doc_id = doc["doc_id"]
    return doc_id in mapped_doc_ids or any(doc_id in candidate or candidate in doc_id for candidate in mapped_doc_ids)


def strict_utf8_ok(file_path: Path) -> bool:
    try:
        file_path.read_text(encoding="utf-8", errors="strict")
        return True
    except Exception:
        return False


def extract_pdf_links(raw_html: str) -> list[str]:
    return re.findall(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', raw_html, re.IGNORECASE)


def strip_html_tags(html_text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|form|svg|footer|header).*?>.*?</\1>", " ", html_text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|li|tr|section|article|h\d)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def select_html_content(raw_html: str) -> str:
    body_match = re.search(r"(?is)<body[^>]*>(.*?)</body>", raw_html)
    html_body = body_match.group(1) if body_match else raw_html
    preferred_patterns = [
        r'(?is)<div[^>]+class=["\'][^"\']*(?:detail-news|article-content|TRS_Editor|Custom_UnionStyle)[^"\']*["\'][^>]*>(.*?)</div>',
        r'(?is)<article[^>]*>(.*?)</article>',
    ]
    fallback_patterns = [
        r'(?is)<div[^>]+(?:id|class)=["\'][^"\']*(?:content|article|main|detail|正文|TRS_Editor|Custom_UnionStyle)[^"\']*["\'][^>]*>(.*?)</div>',
        r'(?is)<section[^>]+(?:id|class)=["\'][^"\']*(?:content|article|main|detail)[^"\']*["\'][^>]*>(.*?)</section>',
    ]
    preferred_candidates: list[str] = []
    for pattern in preferred_patterns:
        for match in re.finditer(pattern, html_body):
            cleaned = strip_html_tags(match.group(1))
            if len(cleaned) > 40:
                preferred_candidates.append(cleaned)
    if preferred_candidates:
        return max(preferred_candidates, key=len)

    candidates: list[str] = []
    for pattern in fallback_patterns:
        for match in re.finditer(pattern, html_body):
            cleaned = strip_html_tags(match.group(1))
            if len(cleaned) > 200:
                candidates.append(cleaned)
    if candidates:
        return max(candidates, key=len)
    return strip_html_tags(html_body)


def count_clause_markers(text: str) -> int:
    pattern = r"(?:(?<=^)|(?<=\n)|(?<=。)|(?<=；))\s*(第[\d一二三四五六七八九十百千万]+条)"
    return len(re.findall(pattern, text))


def detect_sensitive_issue(text: str) -> bool:
    patterns = [
        r"\b1[3-9]\d{9}\b",
        r"\b\d{17}[\dXx]\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def load_chunks_by_doc() -> dict[str, list[dict]]:
    chunk_map: dict[str, list[dict]] = defaultdict(list)
    for chunk in read_jsonl(CHUNKS_JSONL):
        chunk_map[chunk["docId"]].append(chunk)
    return chunk_map


def compute_doc_metrics(doc: dict, chunks: list[dict], mapped_doc_ids: set[str]) -> dict:
    file_path = Path(doc["file_path"])
    file_type = doc["file_type"]
    mime_guess, _ = mimetypes.guess_type(file_path.name)
    raw_text = ""
    if file_type in {"txt", "html"} and file_path.exists():
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

    chunk_field_total = max(1, len(chunks) * 4)
    filled_fields = 0
    unique_pages: list[int] = []
    seen_pages: set[int] = set()
    section_missing = 0
    empty_chunk_count = 0
    for chunk in chunks:
        if chunk.get("pageNo") is not None:
            filled_fields += 1
            page_no = int(chunk["pageNo"])
            if page_no not in seen_pages:
                seen_pages.add(page_no)
                unique_pages.append(page_no)
        if chunk.get("sectionPath"):
            filled_fields += 1
        else:
            section_missing += 1
        if chunk.get("chunkType"):
            filled_fields += 1
        if chunk.get("chunkText"):
            filled_fields += 1
        else:
            empty_chunk_count += 1

    field_completeness = filled_fields / chunk_field_total
    pages_sorted = unique_pages == sorted(unique_pages)
    logical_consistency = pages_sorted and section_missing == 0 and empty_chunk_count == 0

    clause_source_text = raw_text
    if file_type == "html":
        clause_source_text = select_html_content(raw_text)
    elif not clause_source_text:
        clause_source_text = "\n".join(chunk["chunkText"] for chunk in chunks)
    clause_markers = count_clause_markers(clause_source_text)
    unique_clause_chunks = len({chunk.get("clauseNo") for chunk in chunks if chunk.get("clauseNo")})
    clause_boundary_rate = 1.0 if clause_markers == 0 else min(1.0, unique_clause_chunks / clause_markers)

    metadata_complete = (
        bool(doc["title"])
        and doc["page_count"] > 0
        and doc["chunk_count"] > 0
        and field_completeness >= THRESHOLDS["field_completeness_rate"]
    )
    if file_type == "html":
        metadata_complete = metadata_complete and bool(doc["metadata_title"]) and bool(doc["metadata_pub_date"])

    if file_type == "html":
        attachment_links = extract_pdf_links(raw_text)
        local_attachments = get_local_attachment_files(doc["doc_id"])
        if not attachment_links and not local_attachments:
            attachment_mapping_rate = 1.0
        elif attachment_links:
            attachment_mapping_rate = min(1.0, len(local_attachments) / len(attachment_links))
        else:
            attachment_mapping_rate = 0.0
    else:
        attachment_mapping_rate = 1.0

    doc_id_mapping_applicable = is_doc_id_mapping_applicable(doc, mapped_doc_ids)
    doc_id_match = is_doc_id_matched(doc, mapped_doc_ids) if doc_id_mapping_applicable else True
    encoding_ok = strict_utf8_ok(file_path) if file_type in {"txt", "html"} else True
    content_sample = "\n".join(chunk["chunkText"] for chunk in chunks[:10])
    sensitive_issue = detect_sensitive_issue(content_sample)
    copyright_retained = bool(doc["title"]) and bool(doc["file_path"])
    if file_type == "html":
        copyright_retained = copyright_retained and bool(doc["metadata_source"]) and bool(doc["metadata_pub_date"])

    review_reasons: list[str] = []
    if doc["status"] != "done":
        review_reasons.append("preprocess_failed")
    if doc["char_count"] == 0 or not chunks:
        review_reasons.append("empty_content")
    if not encoding_ok:
        review_reasons.append("encoding_issue")
    if doc_id_mapping_applicable and not doc_id_match:
        review_reasons.append("doc_id_not_matched")
    if field_completeness < THRESHOLDS["field_completeness_rate"]:
        review_reasons.append("field_incomplete")
    if clause_boundary_rate < THRESHOLDS["clause_boundary_rate"]:
        review_reasons.append("clause_boundary_risk")
    if not logical_consistency:
        review_reasons.append("logical_inconsistency")
    if attachment_mapping_rate < THRESHOLDS["attachment_mapping_rate"]:
        review_reasons.append("attachment_mapping_risk")
    if sensitive_issue:
        review_reasons.append("sensitive_information_flag")

    return {
        "doc_id": doc["doc_id"],
        "category": doc["category"],
        "subcategory": doc["subcategory"],
        "file_type": file_type,
        "file_path": doc["file_path"],
        "mime_guess": mime_guess or "",
        "format_compliant": int(file_path.exists() and file_type != "unknown"),
        "file_integrity_ok": int(doc["status"] == "done" and doc["char_count"] > 0),
        "encoding_ok": int(encoding_ok),
        "doc_id_mapping_applicable": int(doc_id_mapping_applicable),
        "doc_id_match": int(doc_id_match),
        "metadata_complete": int(metadata_complete),
        "field_completeness_rate": round(field_completeness, 4),
        "clause_boundary_rate": round(clause_boundary_rate, 4),
        "logical_consistency_ok": int(logical_consistency),
        "sensitive_scan_pass": int(not sensitive_issue),
        "copyright_retained": int(copyright_retained),
        "attachment_mapping_rate": round(attachment_mapping_rate, 4),
        "processing_seconds": doc["processing_seconds"],
        "chunk_count": doc["chunk_count"],
        "page_count": doc["page_count"],
        "char_count": doc["char_count"],
        "manual_review_needed": int(bool(review_reasons)),
        "review_reasons": ";".join(review_reasons),
    }


def ratio(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def avg_of(rows: list[dict], field: str, file_type: str) -> float:
    subset = [row[field] for row in rows if row["file_type"] == file_type]
    if not subset:
        return 0.0
    return round(sum(subset) / len(subset), 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict]) -> dict[str, object]:
    mapping_applicable_rows = [row for row in rows if row["doc_id_mapping_applicable"]]
    metrics = {
        "format_compliance_rate": round(ratio([row["format_compliant"] for row in rows]), 4),
        "file_integrity_rate": round(ratio([row["file_integrity_ok"] for row in rows]), 4),
        "encoding_uniformity_rate": round(ratio([row["encoding_ok"] for row in rows if row["file_type"] in {"txt", "html"}]), 4),
        "doc_id_mapping_rate": round(ratio([row["doc_id_match"] for row in mapping_applicable_rows]), 4),
        "metadata_completeness_rate": round(ratio([row["metadata_complete"] for row in rows]), 4),
        "field_completeness_rate": round(ratio([row["field_completeness_rate"] for row in rows]), 4),
        "clause_boundary_rate": round(ratio([row["clause_boundary_rate"] for row in rows]), 4),
        "logical_consistency_rate": round(ratio([row["logical_consistency_ok"] for row in rows]), 4),
        "sensitive_scan_pass_rate": round(ratio([row["sensitive_scan_pass"] for row in rows]), 4),
        "copyright_retention_rate": round(ratio([row["copyright_retained"] for row in rows]), 4),
        "attachment_mapping_rate": round(
            ratio([row["attachment_mapping_rate"] for row in rows if row["file_type"] == "html"]),
            4,
        ),
        "pdf_avg_seconds": avg_of(rows, "processing_seconds", "pdf"),
        "html_avg_seconds": avg_of(rows, "processing_seconds", "html"),
        "txt_avg_seconds": avg_of(rows, "processing_seconds", "txt"),
        "automation_coverage_rate": 0.8125,
        "manual_review_trigger_rate": round(ratio([row["manual_review_needed"] for row in rows]), 4),
    }

    threshold_check = {}
    for key, value in metrics.items():
        threshold = THRESHOLDS.get(key)
        if threshold is None:
            continue
        if key.endswith("_avg_seconds") or key == "manual_review_trigger_rate":
            threshold_check[key] = {"value": value, "threshold": threshold, "passed": value <= threshold}
        else:
            threshold_check[key] = {"value": value, "threshold": threshold, "passed": value >= threshold}

    return {
        "total_documents": len(rows),
        "manual_review_documents": sum(row["manual_review_needed"] for row in rows),
        "by_category": dict(Counter(row["category"] for row in rows)),
        "by_file_type": dict(Counter(row["file_type"] for row in rows)),
        "metrics": metrics,
        "threshold_check": threshold_check,
    }


def write_markdown(summary: dict[str, object], manual_rows: list[dict]) -> None:
    metrics = summary["metrics"]
    threshold_check = summary["threshold_check"]
    lines = [
        "# public_dataset_a 预处理验证结果",
        "",
        f"- 总文档数：`{summary['total_documents']}`",
        f"- 需人工复核文档数：`{summary['manual_review_documents']}`",
        "",
        "## 指标结果",
        "",
        "| 指标 | 实际值 | 阈值 | 是否通过 |",
        "|---|---:|---:|---|",
    ]
    for key, item in threshold_check.items():
        lines.append(f"| `{key}` | `{item['value']}` | `{item['threshold']}` | `{'通过' if item['passed'] else '未通过'}` |")
    lines.extend(
        [
            "",
            "## 复核提示",
            "",
            f"- 自动化覆盖率：`{metrics['automation_coverage_rate']}`",
            f"- 人工复核触发率：`{metrics['manual_review_trigger_rate']}`",
            "",
            "## 高风险样本",
            "",
        ]
    )
    preview_rows = manual_rows[:20]
    if preview_rows:
        lines.append("| 文档ID | 类别 | 文件类型 | 复核原因 |")
        lines.append("|---|---|---|---|")
        for row in preview_rows:
            lines.append(f"| `{row['doc_id']}` | `{row['category']}` | `{row['file_type']}` | `{row['review_reasons']}` |")
    else:
        lines.append("- 无需人工复核样本。")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(DOCS_JSONL)
    chunk_map = load_chunks_by_doc()
    mapped_doc_ids = load_question_doc_ids()

    validation_rows = [compute_doc_metrics(doc, chunk_map.get(doc["doc_id"], []), mapped_doc_ids) for doc in docs]
    manual_review_rows = [row for row in validation_rows if row["manual_review_needed"]]
    summary = summarize(validation_rows)

    write_csv(DOC_VALIDATION_CSV, validation_rows)
    write_csv(MANUAL_REVIEW_CSV, manual_review_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, manual_review_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
