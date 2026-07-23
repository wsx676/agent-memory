"""验证 _format_evidence 优化后关键证据是否不再被丢弃。

直接调用实际的 _format_evidence 函数，对比修改前后效果。
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
from api.services.qwen_client import _format_evidence, _EVIDENCE_MAX_TOTAL, _EVIDENCE_MAX_TOTAL_CHARS
from api.services.retrieval_loop import run_retrieval_loop

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_FILE = PROJECT_ROOT / ".tmp_evidence_fix_verify.txt"

FOCUS_QIDS = ["fin_a_016", "fin_a_019", "fin_a_020"]
OPTION_KEYS = ("A", "B", "C", "D", "E", "F")

# 之前被丢弃的关键chunk
PREVIOUSLY_DROPPED = {
    "fin_a_016": ["annual_midea_2025_report-chunk-103"],
    "fin_a_019": ["annual_catl_2025_report-chunk-7", "annual_catl_2025_report-chunk-59"],
    "fin_a_020": ["annual_catl_2025_report-chunk-5", "annual_catl_2025_report-chunk-59", "annual_catl_2025_report-chunk-60"],
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks_data = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / "chunks_merged.jsonl")]

    questions_file = QUESTIONS_DIR / "financial_reports_questions.json"
    all_questions = json.loads(questions_file.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("_format_evidence 优化后验证")
    lines.append(f"配置: MAX_TOTAL={_EVIDENCE_MAX_TOTAL}, MAX_TOTAL_CHARS={_EVIDENCE_MAX_TOTAL_CHARS}")
    lines.append("=" * 90)

    all_pass = True

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
        candidate_ids = {c.chunk_id for c in candidate_chunks}

        # 调用实际的 _format_evidence
        formatted = _format_evidence(candidate_chunks)

        # 从格式化文本中提取包含的 chunk_id
        included_ids: set[str] = set()
        for chunk in candidate_chunks:
            # _format_evidence 会在文本中包含 location 信息
            # 检查 chunk_id 是否在格式化文本中出现（通过 doc_id + page_no 定位）
            pass

        # 更精确的方法：模拟 _format_evidence 的选择逻辑
        # 但直接用 _format_evidence 的输出更可靠
        # 通过检查格式化文本中是否包含关键chunk的特有内容来判断
        previously_dropped = PREVIOUSLY_DROPPED.get(qid, [])

        lines.append(f"\n{'━' * 90}")
        lines.append(f"题目: {qid}  题型: {answer_format}")
        lines.append(f"候选chunk数: {len(candidate_chunks)}")
        lines.append(f"格式化证据长度: {len(formatted)} 字符")
        lines.append(f"{'━' * 90}")

        # 检查之前被丢弃的关键chunk是否现在被包含
        # 通过检查chunk的特有文本是否在格式化证据中出现
        chunk_map = {c.chunk_id: c for c in candidate_chunks}
        for dropped_id in previously_dropped:
            if dropped_id not in chunk_map:
                lines.append(f"  ⚠️ {dropped_id}: 不在候选列表中（检索未召回）")
                all_pass = False
                continue

            chunk = chunk_map[dropped_id]
            # 取chunk文本中独特的片段来检查
            unique_snippets = []
            text = chunk.chunk_text[:500]

            # 提取含数字的短句作为独特标识
            for sent in re.split(r"[。\n；]", text):
                sent = sent.strip()
                if len(sent) > 10 and re.search(r"\d", sent):
                    unique_snippets.append(sent[:50])

            found = False
            for snippet in unique_snippets[:3]:
                # 对于 prose 类型，_format_evidence 会规范化空白
                normalized_snippet = re.sub(r"\s+", " ", snippet)
                if normalized_snippet in formatted or snippet in formatted:
                    found = True
                    break

            if found:
                lines.append(f"  ✅ {dropped_id}: 已包含在证据中 (page={chunk.page_no}, section={chunk.section_path[:30]})")
            else:
                # 再检查chunk_id对应的location字符串
                loc_pattern = f"doc={chunk.doc_id}, page={chunk.page_no}"
                if loc_pattern in formatted:
                    lines.append(f"  ✅ {dropped_id}: 已包含 (通过location匹配)")
                else:
                    lines.append(f"  ❌ {dropped_id}: 仍被丢弃! (page={chunk.page_no})")
                    lines.append(f"     section: {chunk.section_path}")
                    lines.append(f"     文本预览: {text[:200]}")
                    all_pass = False

        # 统计实际入选/排除
        # 用location模式统计
        included_count = 0
        for chunk in candidate_chunks:
            loc = f"doc={chunk.doc_id}, page={chunk.page_no}"
            if loc in formatted:
                included_count += 1
        excluded_count = len(candidate_chunks) - included_count
        lines.append(f"\n  统计: 入选≈{included_count}/{len(candidate_chunks)}, 排除≈{excluded_count}")

    lines.append(f"\n\n{'=' * 90}")
    if all_pass:
        lines.append("✅ 验证通过: 所有关键chunk均已包含在证据中")
    else:
        lines.append("❌ 验证失败: 部分关键chunk仍被丢弃")
    lines.append("=" * 90)

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Output: {OUTPUT_FILE}")
    print("ALL PASS" if all_pass else "SOME FAILED")


if __name__ == "__main__":
    main()
