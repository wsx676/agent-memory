from __future__ import annotations

import csv
import json
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "public_dataset_a"
RAW_ROOT = DATASET_ROOT / "raw"
QUESTIONS_ROOT = DATASET_ROOT / "questions" / "group_a"
OUTPUT_ROOT = ROOT / "validation_outputs" / "public_dataset_a"
REPORT_JSON = OUTPUT_ROOT / "validation_report.json"
REPORT_CSV = OUTPUT_ROOT / "validation_report.csv"


@dataclass
class ValidationRow:
    category: str
    subcategory: str
    file_path: str
    extension: str
    mime_guess: str
    file_size_bytes: int
    exists: bool
    readable: bool
    content_length: int
    has_title: bool
    has_pubdate: bool
    has_clause: bool
    doc_id_match: bool
    severity: str
    status: str
    notes: str


def load_question_doc_ids() -> set[str]:
    mapped_doc_ids: set[str] = set()
    for question_file in QUESTIONS_ROOT.glob("*.json"):
        data = json.loads(question_file.read_text(encoding="utf-8"))
        for item in data:
            for doc_id in item.get("doc_ids", []):
                mapped_doc_ids.add(str(doc_id))
    return mapped_doc_ids


def iter_raw_files() -> Iterable[Path]:
    for path in RAW_ROOT.rglob("*"):
        if path.is_file():
            yield path


def read_text_best_effort(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".html", ".htm"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore

            document = fitz.open(file_path)
            return "\n".join(page.get_text("text") for page in document)
        except Exception:
            return ""
    return ""


def detect_doc_id(file_path: Path) -> str:
    if file_path.parent.name == "attachments":
        return file_path.stem
    if file_path.parent.name == "html":
        return file_path.stem
    if file_path.parent.name == "txt":
        return file_path.stem
    return file_path.stem


def validate_file(file_path: Path, mapped_doc_ids: set[str]) -> ValidationRow:
    relative_parts = file_path.relative_to(RAW_ROOT).parts
    category = relative_parts[0]
    subcategory = relative_parts[1] if len(relative_parts) > 2 else "root"
    mime_guess, _ = mimetypes.guess_type(file_path.name)
    readable = True
    notes: list[str] = []

    try:
        content = read_text_best_effort(file_path)
    except Exception as exc:  # pragma: no cover
        readable = False
        content = ""
        notes.append(f"read_error={exc}")

    doc_id = detect_doc_id(file_path)
    has_title = bool(re.search(r"<title>|ArticleTitle|^第[\d一二三四五六七八九十百]+条|目录", content, re.MULTILINE))
    has_pubdate = "PubDate" in content or bool(re.search(r"\d{4}-\d{2}-\d{2}", content))
    has_clause = bool(re.search(r"第[\d一二三四五六七八九十百]+条", content))
    doc_id_match = doc_id in mapped_doc_ids or any(doc_id in item for item in mapped_doc_ids)

    severity = "low"
    status = "pass"
    if not readable or file_path.stat().st_size == 0:
        severity = "critical"
        status = "fail"
        notes.append("file_unreadable_or_empty")
    elif len(content.strip()) == 0 and file_path.suffix.lower() in {".pdf", ".txt", ".html", ".htm"}:
        severity = "high"
        status = "fail"
        notes.append("empty_content")
    elif not has_title and file_path.suffix.lower() in {".html", ".txt"}:
        severity = "medium"
        status = "warning"
        notes.append("title_or_structure_missing")
    elif not doc_id_match:
        severity = "medium"
        status = "warning"
        notes.append("doc_id_not_matched")

    return ValidationRow(
        category=category,
        subcategory=subcategory,
        file_path=str(file_path),
        extension=file_path.suffix.lower(),
        mime_guess=mime_guess or "",
        file_size_bytes=file_path.stat().st_size,
        exists=file_path.exists(),
        readable=readable,
        content_length=len(content),
        has_title=has_title,
        has_pubdate=has_pubdate,
        has_clause=has_clause,
        doc_id_match=doc_id_match,
        severity=severity,
        status=status,
        notes=";".join(notes),
    )


def write_outputs(rows: list[ValidationRow]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    REPORT_JSON.write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with REPORT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows: list[ValidationRow]) -> dict[str, object]:
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        category_stats = by_category.setdefault(row.category, {"total": 0, "pass": 0, "warning": 0, "fail": 0})
        category_stats["total"] += 1
        category_stats[row.status] += 1

    return {
        "total_files": len(rows),
        "pass_files": sum(1 for row in rows if row.status == "pass"),
        "warning_files": sum(1 for row in rows if row.status == "warning"),
        "fail_files": sum(1 for row in rows if row.status == "fail"),
        "by_category": by_category,
        "report_json": str(REPORT_JSON),
        "report_csv": str(REPORT_CSV),
    }


def main() -> None:
    mapped_doc_ids = load_question_doc_ids()
    rows = [validate_file(file_path, mapped_doc_ids) for file_path in iter_raw_files()]
    write_outputs(rows)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
