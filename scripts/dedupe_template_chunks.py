"""优先级3：模板文本去重（保守实施）。

背景：金融长文档存在大量无信息量的模板/结构性重复文本——监管处罚书的固定
"复议/诉讼/缴款"模板段、年报标题、纯章节标题、分隔符等。这些切片跨文档/跨页
重复占用 token 预算与检索位次，却不承载答题信息。

策略（与 scripts/_analyze_dup.py 圈定的"安全集合"一致，避免误删）：
仅对以下两类切片的 **完全相同文本**（空白归一化后精确匹配）做全局去重，
每组保留首次出现的一条，删除其余重复：
  A) 结构性 boilerplate：分隔符 / 年报标题(<40字) / 纯章节标题(<30字)
  B) 处罚模板段：出现在处罚类文档中且命中固定模板锚点短语
其余任何切片（含表格、条款正文、数值段）一律不动。

安全保证：
- 不修改原始 chunks_merged.jsonl，产出新文件 chunks_deduped.jsonl
- 去重后校验 docId 覆盖不变（不会把任何文档的切片删空）
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRE = ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
SRC = PRE / "chunks_merged.jsonl"
DST = PRE / "chunks_deduped.jsonl"

PENALTY_MARKERS = ["行政处罚决定书", "行政监管措施决定书", "市场禁入决定书"]
TEMPLATE_ANCHORS = [
    "上述当事人应自收到本处罚决定书之日起", "如不服本处罚决定", "行政复议",
    "行政诉讼", "具体缴款方式", "当事人应于", "将罚没款", "缴纳罚没款",
    "自收到本处罚决定书", "逾期不履行",
]


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_struct_boilerplate(text: str) -> bool:
    n = norm(text)
    if n in ("——", "---", "___", "—", "=", "||"):
        return True
    if re.match(r"^202\d年度报告", n) and len(n) < 40:  # 年报标题
        return True
    if re.match(r"^第[一二三四五六七八九十百千\d]+[章节点]", n) and len(n) < 30:  # 纯章节标题
        return True
    return False


def is_penalty_template(text: str) -> bool:
    return any(a in text for a in TEMPLATE_ANCHORS)


def main() -> None:
    rows = read_rows(SRC)
    penalty_docs = {
        r.get("docId")
        for r in rows
        if any(m in r.get("chunkText", "") for m in PENALTY_MARKERS)
    }

    def is_safe_template(row: dict) -> bool:
        text = row.get("chunkText", "")
        if text.startswith("[表格]"):
            return False  # 表格切片一律保留
        if is_struct_boilerplate(text):
            return True
        if row.get("docId") in penalty_docs and is_penalty_template(text):
            return True
        return False

    seen: set[str] = set()
    kept: list[dict] = []
    removed: list[dict] = []
    for row in rows:
        if is_safe_template(row):
            key = norm(row.get("chunkText", ""))
            if key in seen:
                removed.append(row)
                continue
            seen.add(key)
        kept.append(row)

    # 安全校验：docId 覆盖不得减少
    src_docs = {r.get("docId") for r in rows}
    kept_docs = {r.get("docId") for r in kept}
    lost_docs = src_docs - kept_docs

    with DST.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample = Counter(norm(r.get("chunkText", ""))[:28] for r in removed).most_common(8)
    print(json.dumps({
        "source_chunks": len(rows),
        "kept_chunks": len(kept),
        "removed_chunks": len(removed),
        "removed_pct": round(len(removed) / max(1, len(rows)) * 100, 2),
        "src_doc_count": len(src_docs),
        "kept_doc_count": len(kept_docs),
        "lost_docs": sorted(d for d in lost_docs if d),
        "output": str(DST),
        "removed_samples": sample,
    }, ensure_ascii=False, indent=2))
    if lost_docs:
        print("!! 警告：有文档被删空，需回退！", lost_docs)


if __name__ == "__main__":
    main()
