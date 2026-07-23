"""Pseudo-gold 重跑脚本：用TOC修复后的新chunks重新生成pseudo-gold答案，
与p2v7系统答案和旧pseudo-gold进行三方对比。

用法：
  cd "c:\\Users\\34436\\Downloads\\Agent memory"
  python scripts/pseudo_gold_rerun.py

输出：
  - validation_outputs/public_dataset_a/testing/group_a_qwen_eval/pseudo_gold_new_vs_p2v7.csv
  - 控制台打印对比摘要
"""

from __future__ import annotations

import csv
import json
import re
import sys
import io
import time
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", write_through=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.services.qwen_client import (
    _call_qwen_api,
    _get_setting,
    get_qwen_config_status,
    _option_constraint,
    OPTION_NAMES,
)

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "group_a_qwen_eval"
OLD_GOLD_CSV = PROJECT_ROOT / "60题pseudo-gold总对照表.csv"

OPTION_KEYS = ("A", "B", "C", "D", "E", "F")
MAX_CHARS_PER_DOC = 28000


def load_doc_chunks() -> dict[str, list[dict]]:
    doc_chunks: dict[str, list[dict]] = defaultdict(list)
    with (PREPROCESSED_DIR / "chunks_merged.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            doc_chunks[c["docId"]].append(c)
    return doc_chunks


def load_p2v7() -> dict[str, str]:
    p2v7: dict[str, str] = {}
    path = OUTPUT_DIR / "results__p2v7.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p2v7[r["qid"]] = r["answer"]
    return p2v7


def load_old_gold() -> dict[str, str]:
    old_gold: dict[str, str] = {}
    with OLD_GOLD_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            old_gold[row["qid"]] = row["我的答案"]
    return old_gold


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for fname in ["insurance_questions.json", "regulatory_questions.json", "research_questions.json"]:
        with (QUESTIONS_DIR / fname).open(encoding="utf-8") as f:
            raw = json.load(f)
        for c in raw:
            if isinstance(c.get("options"), dict):
                c["options"] = [str(c["options"][k]) for k in OPTION_KEYS if k in c["options"]]
            cases.append(c)
    return cases


def build_full_text(doc_ids: list[str], doc_chunks: dict[str, list[dict]]) -> str:
    parts: list[str] = []
    for did in doc_ids:
        chunks = doc_chunks.get(did, [])
        if not chunks:
            parts.append(f"=== 文档 {did}（无预处理数据）===\n")
            continue
        chunks.sort(key=lambda c: (c.get("pageNo", 0), c.get("chunkId", "")))
        text_parts: list[str] = []
        total = 0
        for c in chunks:
            ct = c.get("chunkText", "")
            if total + len(ct) > MAX_CHARS_PER_DOC:
                remain = MAX_CHARS_PER_DOC - total
                if remain > 100:
                    text_parts.append(ct[:remain] + "…[截断]")
                break
            text_parts.append(ct)
            total += len(ct)
        full_chars = sum(len(c.get("chunkText", "")) for c in chunks)
        parts.append(
            f"=== 文档 {did}（{len(chunks)}个chunk，{full_chars}字符，"
            f"展示{total}字符）===\n" + "\n".join(text_parts)
        )
    return "\n\n".join(parts)


def build_prompt(question: str, options: list[str], answer_format: str, doc_ids: list[str],
                 doc_chunks: dict[str, list[dict]]) -> list[dict[str, str]]:
    full_text = build_full_text(doc_ids, doc_chunks)
    option_lines = [f"{name}. {opt}" for name, opt in zip(OPTION_NAMES, options)]
    constraint = _option_constraint(options)

    answer_rule = {
        "mcq": f"{constraint}答案必须是单个选项字母。",
        "multi": f"{constraint}答案必须是多个选项字母按字母顺序拼接。",
        "tf": f"{constraint}答案必须是单个选项字母。",
    }.get(answer_format, f"{constraint}答案必须是选项字母。")

    user_prompt = (
        f"以下是与本题相关的文档全文：\n\n{full_text}\n\n"
        f"问题：{question}\n"
        f"选项：\n{chr(10).join(option_lines)}\n\n"
        f"请基于上述文档全文选择答案。逐个选项独立判断："
        "有直接证据支持则选，证据明确排除则不选，无法确认的不选。"
        f"{answer_rule}\n"
        '输出JSON：{"answer":"AC","reasoning":"简述每选项判定理由"}'
    )
    return [
        {"role": "system", "content": "你是金融文档阅读理解助手。只能依据给定文档全文选择答案，不得编造。"},
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    api_key = _get_setting("DASHSCOPE_API_KEY")
    model_name = _get_setting("MODEL_NAME")
    status = get_qwen_config_status()
    if not status.enabled:
        raise RuntimeError("Qwen 未启用")

    print(f"API key={api_key[:20]}... model={model_name}", flush=True)
    print(f"base_url={status.base_url}", flush=True)

    doc_chunks = load_doc_chunks()
    p2v7 = load_p2v7()
    old_gold = load_old_gold()
    cases = load_cases()

    print(f"Loaded {len(cases)} questions, {len(doc_chunks)} docs", flush=True)
    print(f"P2v7 answers: {len(p2v7)}, Old gold: {len(old_gold)}", flush=True)
    print(flush=True)

    results: list[dict] = []

    for i, case in enumerate(cases, 1):
        qid = case["qid"]
        answer_format = case["answer_format"]
        doc_ids = case.get("doc_ids", [])

        full_chars = sum(
            sum(len(c.get("chunkText", "")) for c in doc_chunks.get(did, []))
            for did in doc_ids
        )

        messages = build_prompt(
            case["question"], case["options"], answer_format, doc_ids, doc_chunks
        )

        t0 = time.perf_counter()
        gold_answer = "ERROR"
        tokens = 0
        error_msg = ""
        try:
            result = _call_qwen_api(messages, answer_format, api_key, model_name, status.base_url)
            elapsed = time.perf_counter() - t0
            if result:
                gold_answer = result.answer
                tokens = result.total_tokens
            else:
                gold_answer = "PARSE_FAIL"
                error_msg = "API returned None"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            gold_answer = f"ERR:{type(e).__name__}"
            error_msg = str(e)[:200]

        p2v7_ans = p2v7.get(qid, "?")
        old_ans = old_gold.get(qid, "?")
        match_p2v7 = "✓" if gold_answer == p2v7_ans else "✗"
        match_old = "✓" if gold_answer == old_ans else "✗"

        print(
            f"[{i}/{len(cases)}] {qid} gold={gold_answer} "
            f"p2v7={p2v7_ans}({match_p2v7}) old={old_ans}({match_old}) "
            f"tokens={tokens} {elapsed:.1f}s {error_msg}",
            flush=True,
        )

        results.append({
            "qid": qid,
            "answer_format": answer_format,
            "doc_chars": full_chars,
            "new_gold": gold_answer,
            "p2v7_answer": p2v7_ans,
            "old_gold": old_ans,
            "match_p2v7": match_p2v7,
            "match_old_gold": match_old,
            "tokens": tokens,
            "duration_s": round(elapsed, 1),
        })

    csv_path = OUTPUT_DIR / "pseudo_gold_new_vs_p2v7.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid", "answer_format", "doc_chars",
                "new_gold", "p2v7_answer", "old_gold",
                "match_p2v7", "match_old_gold",
                "tokens", "duration_s",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    match_p2v7_count = sum(1 for r in results if r["match_p2v7"] == "✓")
    match_old_count = sum(1 for r in results if r["match_old_gold"] == "✓")
    total_tokens = sum(r["tokens"] for r in results)
    error_count = sum(1 for r in results if r["new_gold"].startswith("ERR") or r["new_gold"] == "PARSE_FAIL")

    print(f"\n{'=' * 60}")
    print(f"Pseudo-gold 重跑完成")
    print(f"{'=' * 60}")
    print(f"题目数: {len(results)}")
    print(f"错误数: {error_count}")
    print(f"与P2v7一致: {match_p2v7_count}/{len(results)} ({match_p2v7_count / len(results) * 100:.1f}%)")
    print(f"与旧Pseudo-gold一致: {match_old_count}/{len(results)} ({match_old_count / len(results) * 100:.1f}%)")
    print(f"总Token: {total_tokens:,}")
    print(f"输出: {csv_path}")

    print(f"\n--- 与P2v7的差异 ({len(results) - match_p2v7_count}题) ---")
    for r in results:
        if r["match_p2v7"] == "✗":
            print(f"  {r['qid']}: new_gold={r['new_gold']} p2v7={r['p2v7_answer']} old_gold={r['old_gold']}")

    print(f"\n--- 与旧Pseudo-gold的差异 ({len(results) - match_old_count}题) ---")
    for r in results:
        if r["match_old_gold"] == "✗":
            print(f"  {r['qid']}: new_gold={r['new_gold']} old_gold={r['old_gold']} p2v7={r['p2v7_answer']}")


if __name__ == "__main__":
    main()
