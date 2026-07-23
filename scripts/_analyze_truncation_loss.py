"""深入分析被丢弃的chunk是否包含入选chunk中没有的独特证据。

对3个有关键证据丢失的题目（fin_a_016, 019, 020）：
1. 提取被丢弃chunk的完整文本
2. 提取所有入选chunk的合并文本
3. 找出被丢弃chunk中有但入选chunk中没有的独特内容
4. 检查这些独特内容是否是答题关键
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.retrieval_loop import run_retrieval_loop

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_FILE = PROJECT_ROOT / ".tmp_truncation_deep_analysis.txt"

FOCUS_QIDS = ["fin_a_016", "fin_a_019", "fin_a_020"]
OPTION_KEYS = ("A", "B", "C", "D", "E", "F")

_EVIDENCE_MIN_PER_DOC = 5
_EVIDENCE_MAX_TOTAL = 20
_EVIDENCE_SNIPPET_LEN = 1200
_EVIDENCE_TABLE_SNIPPET_LEN = 1200
_EVIDENCE_MAX_TOTAL_CHARS = 18000


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_snippet(chunk: StructuredChunk) -> str:
    if chunk.chunk_type == "table":
        return chunk.chunk_text.strip()[:_EVIDENCE_TABLE_SNIPPET_LEN]
    else:
        return re.sub(r"\s+", " ", chunk.chunk_text).strip()[:_EVIDENCE_SNIPPET_LEN]


def format_evidence_simulation(chunks: list[StructuredChunk]):
    if not chunks:
        return [], [], []

    by_doc: dict[str, list[StructuredChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    selected: list[StructuredChunk] = []
    leftover: list[StructuredChunk] = []
    for doc_chunks in by_doc.values():
        selected.extend(doc_chunks[:_EVIDENCE_MIN_PER_DOC])
        leftover.extend(doc_chunks[_EVIDENCE_MIN_PER_DOC:])

    remaining_slots = _EVIDENCE_MAX_TOTAL - len(selected)
    if remaining_slots > 0:
        selected.extend(leftover[:remaining_slots])
    selected = selected[:_EVIDENCE_MAX_TOTAL]

    excluded_by_count = leftover[remaining_slots:] if remaining_slots > 0 else leftover

    min_required = _EVIDENCE_MIN_PER_DOC * len(by_doc)
    included: list[StructuredChunk] = []
    excluded_by_chars: list[StructuredChunk] = []
    total_chars = 0

    for chunk in selected:
        snippet = get_snippet(chunk)
        location = f"doc={chunk.doc_id}, page={chunk.page_no}"
        if chunk.clause_no:
            location += f", clause={chunk.clause_no}"
        if chunk.section_path:
            location += f", section={chunk.section_path}"
        line = f"[{location}] {snippet}"

        if total_chars + len(line) > _EVIDENCE_MAX_TOTAL_CHARS and len(included) >= min_required:
            excluded_by_chars.append(chunk)
        else:
            included.append(chunk)
            total_chars += len(line)

    return included, excluded_by_count, excluded_by_chars


def extract_numbers(text: str) -> set[str]:
    """提取文本中的数字、百分比、金额等。"""
    numbers = set(re.findall(r"\d+(?:\.\d+)?%?|\d+年|\d+月|\d+日", text))
    return numbers


def find_unique_content(dropped_text: str, included_text: str) -> list[str]:
    """找出被丢弃chunk中有但入选chunk中没有的关键内容。"""
    unique: list[str] = []

    # 1. 独特的数字/百分比
    dropped_numbers = extract_numbers(dropped_text)
    included_numbers = extract_numbers(included_text)
    unique_numbers = dropped_numbers - included_numbers
    if unique_numbers:
        # 过滤掉太短或通用的数字（如年份中的数字片段）
        meaningful = [n for n in unique_numbers if len(n) >= 2 and not (n.endswith("年") and len(n) <= 5)]
        if meaningful:
            unique.append(f"独特数值: {sorted(meaningful)[:20]}")

    # 2. 独特的关键短语（含数字的句子片段）
    dropped_sentences = re.split(r"[。；\n]", dropped_text)
    for sent in dropped_sentences:
        sent = sent.strip()
        if len(sent) < 8:
            continue
        # 检查这个句子是否在入选文本中
        # 取句子的核心部分（含数字的片段）
        if re.search(r"\d", sent) and sent not in included_text:
            # 检查句子的关键部分（去掉空格后）是否在入选文本中
            key_part = re.sub(r"\s+", "", sent)
            included_key = re.sub(r"\s+", "", included_text)
            if key_part[:30] not in included_key:
                unique.append(f"独特语句: {sent[:120]}")

    # 3. 独特的专业术语
    terms = ["连续三年", "50%", "现金分红", "回购", "股权激励", "研发费用", "占比", "比重",
             "经营活动", "现金流量", "净利润", "营业收入", "同比", "环比",
             "分红", "派息", "每10股", "送股", "转增"]
    for term in terms:
        if term in dropped_text and term not in included_text:
            unique.append(f"独特术语: '{term}'")

    return unique


def main() -> None:
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks_data = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / "chunks_merged.jsonl")]

    questions_file = QUESTIONS_DIR / "financial_reports_questions.json"
    all_questions = json.loads(questions_file.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("被丢弃chunk的独特证据深度分析")
    lines.append("=" * 90)

    for raw in all_questions:
        qid = raw["qid"]
        if qid not in FOCUS_QIDS:
            continue

        options_dict = raw.get("options", {})
        options = [options_dict[k] for k in OPTION_KEYS if k in options_dict]
        answer_format = raw["answer_format"]
        doc_ids = raw.get("doc_ids", [])

        request = RunQuestionTaskRequest(
            mode="A",
            qid=qid,
            question=raw["question"],
            options=options,
            answerFormat=answer_format,
            docIds=doc_ids,
        )

        retrieval = run_retrieval_loop(request, available_doc_ids=available_doc_ids, structured_chunks=chunks_data)
        candidate_chunks = retrieval["candidate_chunks"]

        included, excluded_by_count, excluded_by_chars = format_evidence_simulation(candidate_chunks)
        all_excluded = excluded_by_count + excluded_by_chars

        # 入选chunk的合并文本
        included_text = " ".join(get_snippet(c) for c in included)

        lines.append(f"\n{'━' * 90}")
        lines.append(f"题目: {qid}  题型: {answer_format}")
        lines.append(f"选项: {json.dumps(options_dict, ensure_ascii=False)[:200]}")
        lines.append(f"候选: {len(candidate_chunks)}  入选: {len(included)}  被丢弃: {len(all_excluded)}")
        lines.append(f"{'━' * 90}")

        if not all_excluded:
            lines.append("  无被丢弃的chunk")
            continue

        for chunk in all_excluded:
            dropped_text = chunk.chunk_text
            unique_items = find_unique_content(dropped_text, included_text)

            lines.append(f"\n  ── 被丢弃chunk: {chunk.chunk_id}")
            lines.append(f"     doc={chunk.doc_id}, page={chunk.page_no}, type={chunk.chunk_type}")
            lines.append(f"     section={chunk.section_path or '(none)'}")
            lines.append(f"     文本长度={len(dropped_text)}")

            if unique_items:
                lines.append(f"     ⚠️ 独特证据（入选chunk中没有的）:")
                for item in unique_items:
                    lines.append(f"       → {item}")
                # 输出chunk的完整文本（截断到500字符）
                preview = dropped_text[:500].replace("\n", " ")
                lines.append(f"     完整文本预览: {preview}")
            else:
                lines.append(f"     ✅ 无独特证据（内容在入选chunk中已覆盖）")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
