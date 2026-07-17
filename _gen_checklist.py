# -*- coding: utf-8 -*-
"""为 10 道差异题生成'人工核对清单'：按选项关键数字/条款定位到源文档页码+片段。"""
import json, os, re

PRE = r"validation_outputs\public_dataset_a\preprocessed\chunks_merged.jsonl"
OUT = "差异题人工核对清单.md"

# 载入 chunk（按 docId 分组）
docs = {}
with open(PRE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        txt = d.get("chunkText") or ""
        if not txt.strip():
            continue
        docs.setdefault(d["docId"], []).append(d)

def find(docs_filter, terms, max_hits=3):
    """在指定 docId 集合中搜索 terms，返回命中 [(docId,pageNo,snippet)]"""
    hits = []
    for did, chunks in docs.items():
        if docs_filter and did not in docs_filter:
            continue
        for c in chunks:
            t = c.get("chunkText") or ""
            low = t.lower()
            for term in terms:
                if term.lower() in low:
                    i = low.find(term.lower())
                    s = max(0, i - 60)
                    e = min(len(t), i + len(term) + 60)
                    snippet = re.sub(r"\s+", " ", t[s:e]).strip()
                    hits.append((did, c.get("pageNo"), term, snippet))
                    break
    # 去重并按页码排序，限制数量
    seen = set()
    uniq = []
    for h in sorted(hits, key=lambda x: (x[0], x[1] if x[1] is not None else 0)):
        key = (h[0], h[1], h[3])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq[:max_hits]

# (qid, doc_ids, [(选项说明, [搜索词])])
QUESTIONS = [
    ("fc_a_005", {"text01", "text10"}, [
        ("A. 发行人=广东省广晟控股集团", ["广东省广晟控股集团"]),
        ("B. 发行股份购买资产+募集配套资金(科源制药)", ["科源制药", "发行股份购买资产", "募集配套资金"]),
        ("C. 资产负债率68.06%(text01)", ["68.06%", "合并口径资产负债率"]),
        ("D. 力诺投资资产负债率43.24%", ["力诺投资", "43.24%"]),
    ]),
    ("fc_a_008", {"text02", "text03"}, [
        ("A. 发行金额10亿元", ["10亿元", "发行金额", "发行规模"]),
        ("B. 违约赔偿150%惩罚系数", ["150%", "违约赔偿", "惩罚系数"]),
        ("C. 主体信用评级AAA", ["AAA", "主体信用", "信用评级"]),
        ("D. 发行人=厦门金圆投资集团", ["厦门金圆", "厦门金圆投资"]),
    ]),
    ("fin_a_005", {"annual_byd_2024_report", "annual_midea_2024_report"}, [
        ("A. 比亚迪营收>美的(2024)", ["营业收入", "营业总收入"]),
        ("B. 比亚迪研发投入强度>美的", ["研发投入", "研发投入占"]),
        ("C. 经营活动现金流净额(美的>比亚迪)", ["经营活动产生的现金流量净额", "经营活动现金流净额"]),
        ("D. 归母净利润双位数同比增长(两家)", ["归属于上市公司股东的净利润", "净利润同比", "同比增长"]),
    ]),
    ("fin_a_009", {"annual_byd_2024_report", "annual_chinamobile_2025_report"}, [
        ("A. 中国移动营收>1万亿(10502亿)", ["10,502", "10502", "营业收入", "万亿元"]),
        ("B. 研发占比:比亚迪>中国移动", ["研发投入占", "研发投入"]),
        ("C. 经营活动现金流净额均为正", ["经营活动产生的现金流量净额", "经营活动现金流净额"]),
        ("D. 中国移动净利润同比-0.9%/2.0%", ["归属于母公司股东的净利润", "同比下降0.9%", "同比增长2.0%", "同比0.9%"]),
    ]),
    ("fin_a_011", {"annual_byd_2025_report", "annual_midea_2025_report"}, [
        ("A. 美的归母净利润增速>比亚迪", ["归属于上市公司股东的净利润", "净利润", "同比增长"]),
        ("B. 比亚迪研发占比<美的", ["研发投入占", "研发投入"]),
        ("C. 每股现金分红:美的>比亚迪", ["每股现金分红", "每10股", "现金分红"]),
        ("D. 经营活动现金流净额均为正(2025)", ["经营活动产生的现金流量净额", "经营活动现金流净额"]),
    ]),
    ("fin_a_015", {"annual_catl_2024_report", "annual_chinamobile_2025_report"}, [
        ("A. 宁德时代现金分红占归母净利润20%", ["现金分红", "利润分配", "20%", "分红比例"]),
        ("B. 中国移动营收同比-0.9%(实为+0.9%)", ["营业收入", "同比增长0.9%", "同比0.9%", "下降0.9%"]),
        ("C. 宁德时代每股分红69.57元/0.4553元", ["每股分红", "69.57", "0.4553", "每10股"]),
        ("D. 中国移动研发费用占营收>5%(实为2.8%)", ["研发费用", "研发投入", "占营业收入", "2.8%"]),
    ]),
    ("ins_a_007", {"1", "2", "16"}, [
        ("A. 平安智盈金生保单贷款80%现金价值", ["智盈金生", "保单贷款", "现金价值"]),
        ("B. 国寿增益宝保单贷款(扣欠款后80%)第二十二条", ["增益宝", "第二十二条", "保单贷款", "现金价值扣除"]),
        ("C. 平安富鸿金生个人养老金不允许贷款(6.2)", ["富鸿金生", "6.2", "个人养老金", "保单贷款"]),
        ("D. 富鸿金生无论何种都不允许贷款", ["富鸿金生", "保单贷款"]),
    ]),
    ("ins_a_015", {"1", "4", "7", "13"}, [
        ("A. 平安智盈金生自杀2年内免责", ["智盈金生", "自杀", "2年"]),
        ("B. 平安安佑福故意自伤/自杀2年内免责", ["安佑福", "故意自伤", "自杀"]),
        ("C. 平安预防接种意外险自杀免责无2年例外", ["预防接种", "意外险", "自杀"]),
        ("D. 众安食责险不涉及自杀", ["众安", "食责险", "自杀"]),
    ]),
    ("res_a_016", {"pack2_text03", "pack2_text20"}, [
        ("A. 2025.12服务零售累计同比>商品零售", ["服务零售", "商品零售", "累计同比"]),
        ("B. 上市银行手续费及佣金净收入负增长", ["手续费及佣金净收入", "手续费及佣金"]),
        ("C. 居民可支配收入6.33%→4.99%(2023-2025)", ["6.33%", "4.99%", "可支配收入"]),
        ("D. 上市险企四季度利润承压(资本市场震荡)", ["上市险企", "四季度", "利润承压", "资本市场震荡"]),
    ]),
    ("res_a_017", {"pack2_text03", "pack2_text17"}, [
        ("A. 2025金融信创市场≈2500亿元", ["金融信创", "2500亿", "信创市场"]),
        ("B. 服务消费占比低位(高储蓄)", ["服务消费占比", "高储蓄", "服务消费"]),
        ("C. 天阳科技2025净利润同比", ["天阳科技", "净利润", "同比"]),
        ("D. 居民收入增速2023-2025持续放缓", ["居民收入", "6.33%", "4.99%", "放缓"]),
    ]),
]

lines = ["# 差异题人工核对清单（源文档精确页码定位）", "",
         "> 页码来自预处理 chunk 的 `pageNo` 字段，对应你打开的原始 PDF 页面。",
         "> 每题列出各选项关键证据所在页码+片段，供你翻到该页人工核实。", ""]

for qid, dids, opts in QUESTIONS:
    lines.append(f"## {qid}")
    lines.append("")
    lines.append(f"**源文档**：{', '.join(sorted(dids))}")
    lines.append("")
    for label, terms in opts:
        lines.append(f"- **{label}**")
        res = find(dids, terms)
        if not res:
            lines.append(f"  - ⚠ 未在 chunk 中匹配到 `{terms}`（可能 OCR 未识别或需翻目录）")
        else:
            for did, page, term, snip in res:
                pg = page if page is not None else "?"
                lines.append(f"  - `p{pg}`（{did}）命中「{term}」：…{snip}…")
    lines.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written:", OUT)
