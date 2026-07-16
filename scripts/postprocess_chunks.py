"""Post-process preprocessed chunks: merge fragments, annotate formulas, quality check.

Usage:
    python scripts/postprocess_chunks.py

Reads:  validation_outputs/public_dataset_a/preprocessed/chunks.jsonl
Writes: validation_outputs/public_dataset_a/preprocessed/chunks_merged.jsonl
        validation_outputs/public_dataset_a/preprocessed/merge_report.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
INPUT_FILE = PREPROCESSED_DIR / "chunks.jsonl"
OUTPUT_FILE = PREPROCESSED_DIR / "chunks_merged.jsonl"
REPORT_FILE = PREPROCESSED_DIR / "merge_report.json"

# ── Merge parameters ──────────────────────────────────────────────
# 适配新预处理产物（中位数 295 字）：降低合并门槛，让更多中等长度切片
# 能与短切片合并。原 TARGET_MIN=300 适配旧碎片产物（中位数 91 字），
# 对新产物几乎不触发合并。
TARGET_MIN = 200   # Chunks below this are merge candidates
TARGET_MAX = 800   # Don't exceed this when merging
ABSOLUTE_MAX = 1200  # Hard cap (matches preprocess.py truncation)

# ── Noise cleanup patterns ────────────────────────────────────────
# HTML 页面残留的导航/功能链接文本，在容器提取阶段未被完全剔除。
NOISE_PATTERNS = [
    re.compile(r"【打印】"),
    re.compile(r"【关闭窗口】"),
    re.compile(r"【收藏】"),
    re.compile(r"【分享】"),
    re.compile(r"链接：\s*$"),  # 页脚"链接："引导词
]

# ── Formula annotation patterns ───────────────────────────────────
FORMULA_PATTERNS = [
    # Percentage: 75%, 12.5%
    (re.compile(r"\d+(?:\.\d+)?\s*%"), "[公式] 百分比"),
    # Ratio/multiplier: 3倍, 1.5倍
    (re.compile(r"\d+(?:\.\d+)?\s*倍"), "[公式] 倍数"),
    # Amount: 万元, 亿元, 元
    (re.compile(r"\d+(?:\.\d+)?\s*(?:万元|亿元|元)"), "[公式] 金额"),
    # Division: A/B or A÷B
    (re.compile(r"\d+(?:\.\d+)?\s*[÷/]\s*\d+(?:\.\d+)?"), "[公式] 除法"),
    # Formula keywords
    (re.compile(r"(?:计算公式|等于|比例|乘以|除以|计算方式|计算方法)"), "[公式] 计算表述"),
]

# ── Quality check patterns ────────────────────────────────────────
GARBLED_PATTERN = re.compile(r"[\ufffd]")  # Unicode replacement char
LONG_NON_CJK_PATTERN = re.compile(r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]{50,}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_chunk_index(chunk_id: str) -> int:
    """Extract numeric index from chunkId like 'docId-chunk-42'."""
    m = re.search(r"chunk-(\d+)$", chunk_id or "")
    return int(m.group(1)) if m else 0


def should_merge(a: dict, b: dict) -> bool:
    """Decide if two adjacent chunks should be merged."""
    # Never merge across documents
    if a["docId"] != b["docId"]:
        return False
    # Never merge table chunks with non-table
    if a.get("chunkType") == "table" or b.get("chunkType") == "table":
        return False
    # Only merge same or adjacent pages
    page_diff = abs((a.get("pageNo", 0)) - (b.get("pageNo", 0)))
    if page_diff > 1:
        return False
    # Only merge if same section path (or one is empty)
    sec_a = a.get("sectionPath", "") or ""
    sec_b = b.get("sectionPath", "") or ""
    if sec_a and sec_b and sec_a != sec_b:
        return False
    # Don't merge if combined would exceed target
    combined_len = len(a.get("chunkText", "")) + len(b.get("chunkText", ""))
    if combined_len > ABSOLUTE_MAX:
        return False
    # At least one should be small
    if len(a.get("chunkText", "")) >= TARGET_MIN and len(b.get("chunkText", "")) >= TARGET_MIN:
        return False
    # Don't merge if second chunk starts with a clause boundary
    text_b = b.get("chunkText", "").strip()
    if re.match(r"^第[\d一二三四五六七八九十百千万]+[条章节]", text_b):
        return False
    return True


def merge_two(a: dict, b: dict) -> dict:
    """Merge two chunks into one, preserving metadata."""
    combined_text = a.get("chunkText", "") + "\n" + b.get("chunkText", "")
    combined_text = combined_text[:ABSOLUTE_MAX]

    # Use the first non-null clause number
    clause = a.get("clauseNo") or b.get("clauseNo")

    # Use the more specific section path
    sec = a.get("sectionPath", "") or b.get("sectionPath", "")

    # Use the smaller page number (start of merged range)
    page = min(a.get("pageNo", 1), b.get("pageNo", 1))

    # Chunk type: if either is clause, result is clause
    ctype = "clause" if a.get("chunkType") == "clause" or b.get("chunkType") == "clause" else (a.get("chunkType") or b.get("chunkType") or "prose")

    return {
        "chunkId": a["chunkId"],  # Keep first chunk's ID
        "docId": a["docId"],
        "title": a.get("title", ""),
        "domain": a.get("domain", ""),
        "pageNo": page,
        "sectionPath": sec,
        "chunkType": ctype,
        "clauseNo": clause,
        "chunkText": combined_text,
    }


def merge_chunks(chunks: list[dict]) -> list[dict]:
    """Merge adjacent small chunks within each document."""
    # Group by docId
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["docId"], []).append(c)

    merged_total: list[dict] = []
    merge_count = 0

    for doc_id, doc_chunks in by_doc.items():
        # Sort by chunk index to preserve order
        doc_chunks.sort(key=lambda c: extract_chunk_index(c.get("chunkId", "")))

        result: list[dict] = []
        for chunk in doc_chunks:
            if result and should_merge(result[-1], chunk):
                result[-1] = merge_two(result[-1], chunk)
                merge_count += 1
            else:
                result.append(dict(chunk))  # copy
        merged_total.extend(result)

    return merged_total, merge_count


def clean_noise(chunks: list[dict]) -> int:
    """Remove residual HTML navigation noise from chunk text.

    Strips patterns like 【打印】【关闭窗口】 that leak through container
    extraction. Returns count of chunks modified.
    """
    cleaned_count = 0
    for chunk in chunks:
        text = chunk.get("chunkText", "")
        if not text:
            continue
        original = text
        for pattern in NOISE_PATTERNS:
            text = pattern.sub("", text)
        # Collapse whitespace left by removals
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if text != original:
            chunk["chunkText"] = text
            cleaned_count += 1
    return cleaned_count


def annotate_formulas(chunks: list[dict]) -> int:
    """Add [公式] annotations to chunks containing formula patterns. Returns count of annotated chunks."""
    annotated = 0
    for chunk in chunks:
        text = chunk.get("chunkText", "")
        tags: list[str] = []
        for pattern, label in FORMULA_PATTERNS:
            if pattern.search(text) and label not in tags:
                tags.append(label)
        if tags:
            # Only add annotation if not already present
            prefix = " ".join(tags[:3])  # max 3 tags
            if not text.startswith("[公式]"):
                chunk["chunkText"] = f"{prefix} {text[:ABSOLUTE_MAX - len(prefix) - 1]}"
                annotated += 1
    return annotated


def quality_check(chunks: list[dict]) -> dict[str, Any]:
    """Run quality checks on chunks and return report."""
    issues: dict[str, list[str]] = {
        "garbled_chars": [],
        "long_non_cjk": [],
        "empty_text": [],
        "no_section_path": [],
        "excessively_short": [],
    }

    for c in chunks:
        cid = c.get("chunkId", "?")
        text = c.get("chunkText", "")

        if not text.strip():
            issues["empty_text"].append(cid)
        if GARBLED_PATTERN.search(text):
            issues["garbled_chars"].append(cid)
        if LONG_NON_CJK_PATTERN.search(text):
            issues["long_non_cjk"].append(cid)
        if not (c.get("sectionPath") or "").strip():
            issues["no_section_path"].append(cid)
        if len(text.strip()) < 20 and c.get("chunkType") != "table":
            issues["excessively_short"].append(cid)

    lengths = [len(c.get("chunkText", "")) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 1) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "distribution": {
            "lt_200": sum(1 for l in lengths if l < 200),
            "lt_200_pct": round(sum(1 for l in lengths if l < 200) / len(lengths) * 100, 1) if lengths else 0,
            "range_200_500": sum(1 for l in lengths if 200 <= l < 500),
            "range_200_500_pct": round(sum(1 for l in lengths if 200 <= l < 500) / len(lengths) * 100, 1) if lengths else 0,
            "range_500_800": sum(1 for l in lengths if 500 <= l < 800),
            "range_500_800_pct": round(sum(1 for l in lengths if 500 <= l < 800) / len(lengths) * 100, 1) if lengths else 0,
            "gte_800": sum(1 for l in lengths if l >= 800),
            "gte_800_pct": round(sum(1 for l in lengths if l >= 800) / len(lengths) * 100, 1) if lengths else 0,
        },
        "type_distribution": dict(Counter(c.get("chunkType", "?") for c in chunks)),
        "clause_coverage": {
            "with_clause": sum(1 for c in chunks if c.get("clauseNo")),
            "pct": round(sum(1 for c in chunks if c.get("clauseNo")) / len(chunks) * 100, 1) if chunks else 0,
        },
        "quality_issues": {k: len(v) for k, v in issues.items()},
        "issue_samples": {k: v[:10] for k, v in issues.items()},
    }


def main() -> None:
    print("Loading chunks...")
    original = read_jsonl(INPUT_FILE)
    print(f"  Original: {len(original)} chunks")

    # Step 0: Clean HTML navigation noise (before merge so lengths are accurate)
    print("Cleaning noise...")
    noise_count = clean_noise(original)
    print(f"  Cleaned: {noise_count} chunks had noise removed")

    # Pre-merge quality report
    pre_report = quality_check(original)
    print(f"  Pre-merge: median={pre_report['length_stats']['median']}, <200chars={pre_report['distribution']['lt_200_pct']}%")

    # Step 1: Merge small chunks
    print("Merging small chunks...")
    merged, merge_count = merge_chunks(original)
    print(f"  Merged: {len(merged)} chunks (removed {len(original) - len(merged)} via {merge_count} merges)")

    # Step 2: Annotate formulas
    print("Annotating formulas...")
    formula_count = annotate_formulas(merged)
    print(f"  Annotated: {formula_count} chunks with formula tags")

    # Post-merge quality report
    post_report = quality_check(merged)
    print(f"  Post-merge: median={post_report['length_stats']['median']}, <200chars={post_report['distribution']['lt_200_pct']}%")

    # Write output
    write_jsonl(OUTPUT_FILE, merged)
    print(f"  Written: {OUTPUT_FILE}")

    # Write report
    report = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "original_count": len(original),
        "merged_count": len(merged),
        "merge_operations": merge_count,
        "noise_cleaned": noise_count,
        "formula_annotations": formula_count,
        "pre_merge": pre_report,
        "post_merge": post_report,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Report: {REPORT_FILE}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
