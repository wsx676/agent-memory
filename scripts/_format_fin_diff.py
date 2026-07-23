"""将 .tmp_fin_diff_chunks.jsonl 转为可读文本文件，每题一个。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / ".tmp_fin_diff_chunks.jsonl"

with JSONL.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = r["qid"]
        out_path = ROOT / f".tmp_fin_diff_{qid}.txt"
        with out_path.open("w", encoding="utf-8") as out:
            out.write(f"=== {qid} ===\n")
            out.write(f"题型: {r['answer_format']} / {r['type']}\n")
            out.write(f"问题: {r['question']}\n")
            out.write(f"文档: {', '.join(r['doc_ids'])}\n")
            out.write(f"召回chunks数: {r['chunk_count']}\n")
            out.write(f"候选文档: {', '.join(r.get('candidate_doc_ids', []))}\n")
            out.write("\n选项:\n")
            for k, v in r["options"].items():
                out.write(f"  {k}: {v}\n")
            out.write("\n--- 召回片段 ---\n\n")
            for j, c in enumerate(r["chunks"], 1):
                out.write(f"[Chunk {j}] doc={c['doc_id']} page={c['page_no']} type={c['chunk_type']}\n")
                out.write(f"section: {c['section_path']}\n")
                out.write(c["chunk_text"] + "\n\n")
        print(f"  {qid} -> {out_path.name}")
print("Done")
