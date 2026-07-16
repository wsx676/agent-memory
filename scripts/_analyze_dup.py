"""临时分析：测算安全的精确去重规模（处罚模板 + 结构性 boilerplate）。"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
rows = [json.loads(l) for l in (PRE / "chunks_merged.jsonl").open(encoding="utf-8") if l.strip()]

penalty_markers = ["行政处罚决定书", "行政监管措施决定书", "市场禁入决定书"]
penalty_docs = {r.get("docId") for r in rows if any(m in r.get("chunkText", "") for m in penalty_markers)}

template_anchors = ["上述当事人应自收到本处罚决定书之日起", "如不服本处罚决定", "行政复议",
                   "行政诉讼", "具体缴款方式", "当事人应于", "将罚没款", "缴纳罚没款",
                   "自收到本处罚决定书", "逾期不履行"]

def norm(t):
    return re.sub(r"\s+", "", t).strip()

def is_struct_boilerplate(t):
    n = norm(t)
    if n in ("——", "---", "___", "—", "=", "||"):
        return True
    if re.match(r"^202\d年度报告", n) and len(n) < 40:   # 年报标题
        return True
    if re.match(r"^第[一二三四五六七八九十百千\d]+[章节点节]", n) and len(n) < 30:  # 纯章节标题
        return True
    return False

def is_penalty_template(t):
    return any(a in t for a in template_anchors)

# A) 仅处罚模板（精确）
pt = [r for r in rows if r.get("docId") in penalty_docs and is_penalty_template(r.get("chunkText", ""))]
c = Counter(norm(r["chunkText"]) for r in pt)
print("处罚模板 精确可删:", sum(v-1 for v in c.values() if v > 1), " (组", len([1 for v in c.values() if v>1]), ")")

# B) 结构性 boilerplate（精确）
sb = [r for r in rows if is_struct_boilerplate(r.get("chunkText", ""))]
c = Counter(norm(r["chunkText"]) for r in sb)
print("结构性boilerplate 精确可删:", sum(v-1 for v in c.values() if v > 1), " (组", len([1 for v in c.values() if v>1]), ")")
print("  结构性 boilerplate 总数:", len(sb), " 样例:", Counter(norm(r['chunkText'])[:30] for r in sb).most_common(6))

# C) 两者并集（精确）
union = [r for r in rows if (r.get("docId") in penalty_docs and is_penalty_template(r.get("chunkText", ""))) or is_struct_boilerplate(r.get("chunkText", ""))]
c = Counter(norm(r["chunkText"]) for r in union)
print("并集 精确可删:", sum(v-1 for v in c.values() if v > 1), " (组", len([1 for v in c.values() if v>1]), ")")

# D) 全局精确重复（任意 chunk），看上限
c = Counter(norm(r["chunkText"]) for r in rows)
print("全局 精确可删(任意):", sum(v-1 for v in c.values() if v > 1))
