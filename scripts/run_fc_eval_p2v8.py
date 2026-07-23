#!/usr/bin/env python
"""Run 20 fc questions through the improved pipeline (p2v8), record
detailed trajectory, and compare with p2v7 + pseudo-gold.

Usage:
    python scripts/run_fc_eval_p2v8.py [--suffix _p2v10_nothink]
"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import RunQuestionTaskRequest, StructuredChunk
from api.services.formatter import (
    build_evidence_items,
    needs_self_check,
)
from api.services.qwen_client import (
    QwenAnswer,
    answer_with_qwen,
    get_qwen_config_status,
    verify_with_qwen,
)
from api.services.retrieval_loop import run_retrieval_loop

QUESTIONS_PATH = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a" / "financial_contracts_questions.json"
PREPROCESSED_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "group_a_qwen_eval"

OPTION_KEYS = ("A", "B", "C", "D", "E", "F")

# Pseudo-gold answers reconstructed from manual document analysis
PSEUDO_GOLD = {
    "fc_a_001": "ABD",
    "fc_a_002": "ABD",
    "fc_a_003": "A",
    "fc_a_004": "AD",
    "fc_a_005": "ABD",
    "fc_a_006": "A",
    "fc_a_007": "BD",
    "fc_a_008": "B",
    "fc_a_009": "ABD",
    "fc_a_010": "A",
    "fc_a_011": "AB",
    "fc_a_012": "ACD",
    "fc_a_013": "A",
    "fc_a_014": "AB",
    "fc_a_015": "B",
    "fc_a_016": "ABC",
    "fc_a_017": "ABD",
    "fc_a_018": "A",
    "fc_a_019": "AB",
    "fc_a_020": "AB",
}

# p2v7 answers (from previous run, before improvements)
P2V7_ANSWERS = {
    "fc_a_001": "ABD",
    "fc_a_002": "AD",
    "fc_a_003": "A",
    "fc_a_004": "D",
    "fc_a_005": "ABD",
    "fc_a_006": "A",
    "fc_a_007": "BCD",
    "fc_a_008": "C",
    "fc_a_009": "ABD",
    "fc_a_010": "A",
    "fc_a_011": "ABD",
    "fc_a_012": "B",
    "fc_a_013": "A",
    "fc_a_014": "A",
    "fc_a_015": "B",
    "fc_a_016": "BC",
    "fc_a_017": "ABD",
    "fc_a_018": "B",
    "fc_a_019": "AD",
    "fc_a_020": "ABC",
}


def normalize_options(raw_options: Any) -> list[str] | None:
    if isinstance(raw_options, dict):
        return [str(raw_options[k]) for k in OPTION_KEYS if k in raw_options] or None
    if isinstance(raw_options, list):
        return [str(i) for i in raw_options] or None
    return None


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def chunk_summary(chunk: StructuredChunk) -> dict:
    """Compact chunk info for trajectory logging."""
    snippet = re.sub(r"\s+", " ", chunk.chunk_text or "").strip()[:120]
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page_no": chunk.page_no,
        "section_path": chunk.section_path or "",
        "chunk_type": chunk.chunk_type,
        "snippet": snippet,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="Output file suffix, e.g. _p2v10_nothink")
    args = parser.parse_args()
    suffix = args.suffix

    qwen_status = get_qwen_config_status()
    if not qwen_status.enabled:
        raise RuntimeError(f"Qwen not enabled: {qwen_status.disable_reason}")
    print(f"Qwen enabled (model={qwen_status.requested_model}), starting evaluation...suffix={suffix or '(none)'}")

    # Load data
    documents = read_jsonl(PREPROCESSED_DIR / "documents.jsonl")
    available_doc_ids = {item["doc_id"] for item in documents if item.get("status", "done") == "done"}
    chunks = [StructuredChunk.model_validate(item) for item in read_jsonl(PREPROCESSED_DIR / "chunks_merged.jsonl")]
    print(f"Loaded {len(chunks)} chunks, {len(available_doc_ids)} doc_ids")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} fc questions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / f"results__p2v8{suffix}.jsonl"
    trajectory_path = OUTPUT_DIR / f"trajectory__p2v8{suffix}.jsonl"

    results: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for index, q in enumerate(questions, start=1):
        qid = q["qid"]
        started = time.perf_counter()

        options = normalize_options(q.get("options"))
        request = RunQuestionTaskRequest(
            mode="A",
            qid=qid,
            question=q["question"],
            options=options,
            answerFormat=q["answer_format"],
            docIds=q["doc_ids"],
        )

        # --- Retrieval ---
        retrieval = run_retrieval_loop(
            request,
            available_doc_ids=available_doc_ids,
            structured_chunks=chunks,
        )
        candidate_chunks = retrieval["candidate_chunks"]
        reasoning_results = retrieval["reasoning_results"]

        # --- LLM answer ---
        qwen_result = None
        qwen_error = None
        try:
            qwen_result = answer_with_qwen(
                question=request.question,
                options=request.options,
                answer_format=request.answer_format,
                evidence=list(candidate_chunks),
                reasoning_hints=list(reasoning_results),
                force_thinking=(request.answer_format != "tf"),
            )
        except Exception as exc:
            qwen_error = f"{type(exc).__name__}: {exc}"

        llm_answer = qwen_result.answer if qwen_result is not None else "A"
        llm_before_selfcheck = llm_answer

        # --- Self-check ---
        self_checked = False
        verify_error = None
        if qwen_result is not None and needs_self_check(
            q["answer_format"], list(reasoning_results), llm_answer,
        ):
            self_checked = True
            try:
                verified = verify_with_qwen(
                    question=request.question,
                    options=request.options,
                    answer_format=request.answer_format,
                    evidence=list(candidate_chunks),
                    reasoning_hints=list(reasoning_results),
                    first_answer=llm_answer,
                )
                if verified is not None:
                    llm_answer = verified.answer
                    qwen_result = QwenAnswer(
                        answer=llm_answer,
                        reasoning=verified.reasoning,
                        prompt_tokens=qwen_result.prompt_tokens + verified.prompt_tokens,
                        completion_tokens=qwen_result.completion_tokens + verified.completion_tokens,
                        total_tokens=qwen_result.total_tokens + verified.total_tokens,
                        model=verified.model,
                    )
            except Exception as exc:
                verify_error = f"{type(exc).__name__}: {exc}"

        answer_after_selfcheck = llm_answer

        final_answer = llm_answer
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        pseudo_gold = PSEUDO_GOLD.get(qid, "")
        p2v7 = P2V7_ANSWERS.get(qid, "")
        matches_gold = final_answer == pseudo_gold
        matches_p2v7 = final_answer == p2v7

        # --- Result row ---
        row = {
            "qid": qid,
            "domain": q.get("domain", "financial_contracts"),
            "answer_format": q["answer_format"],
            "answer": final_answer,
            "llm_before_selfcheck": llm_before_selfcheck,
            "answer_after_selfcheck": answer_after_selfcheck,
            "self_checked": self_checked,
            "pseudo_gold": pseudo_gold,
            "p2v7_answer": p2v7,
            "matches_gold": matches_gold,
            "matches_p2v7": matches_p2v7,
            "prompt_tokens": int(qwen_result.prompt_tokens) if qwen_result else 0,
            "completion_tokens": int(qwen_result.completion_tokens) if qwen_result else 0,
            "total_tokens": int(qwen_result.total_tokens) if qwen_result else 0,
            "duration_ms": duration_ms,
            "candidate_chunk_count": len(candidate_chunks),
            "evidence_count": len(build_evidence_items(reasoning_results, retrieval["evidence_map"])),
            "qwen_error": qwen_error,
            "verify_error": verify_error,
        }
        results.append(row)

        # --- Trajectory ---
        traj = {
            "qid": qid,
            "question": q["question"],
            "options": q["options"],
            "doc_ids": q["doc_ids"],
            "answer_format": q["answer_format"],
            "final_answer": final_answer,
            "pseudo_gold": pseudo_gold,
            "matches_gold": matches_gold,
            "p2v7_answer": p2v7,
            "matches_p2v7": matches_p2v7,
            "llm_before_selfcheck": llm_before_selfcheck,
            "answer_after_selfcheck": answer_after_selfcheck,
            "self_checked": self_checked,
            "candidate_chunks": [chunk_summary(c) for c in candidate_chunks],
            "reasoning_results": [
                {
                    "option": r.option,
                    "verdict": r.verdict,
                    "reasoning": (r.reasoning or "")[:200],
                }
                for r in reasoning_results
            ],
            "retrieval_quality": retrieval.get("evaluation", {}).get("quality_label", ""),
            "loop_logs": retrieval.get("loop_logs", []),
            "total_tokens": row["total_tokens"],
            "duration_ms": duration_ms,
        }
        trajectories.append(traj)

        status = "GOLD" if matches_gold else "MISS"
        p2v7_change = "=p2v7" if matches_p2v7 else "CHANGED"
        print(
            f"  [{index:2d}/20] {qid}: ans={final_answer} gold={pseudo_gold} "
            f"p2v7={p2v7} [{status}] [{p2v7_change}] "
            f"sc={self_checked} "
            f"chunks={len(candidate_chunks)} tokens={row['total_tokens']}"
        )

    # --- Save results ---
    with open(results_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {results_path}")

    with open(trajectory_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
    print(f"Trajectory saved to {trajectory_path}")

    # --- Summary ---
    gold_matches = sum(1 for r in results if r["matches_gold"])
    p2v7_matches = sum(1 for r in results if r["matches_p2v7"])
    total_tokens = sum(r["total_tokens"] for r in results)
    changed = sum(1 for r in results if not r["matches_p2v7"])

    print("\n" + "=" * 80)
    print("SUMMARY: p2v8 vs Pseudo-gold vs p2v7")
    print("=" * 80)
    print(f"  Pseudo-gold match: {gold_matches}/20 ({100*gold_matches/20:.0f}%)")
    print(f"  p2v7 match:        {p2v7_matches}/20 ({100*p2v7_matches/20:.0f}%)  [p2v7 vs gold was 9/20=45%]")
    print(f"  Changed from p2v7: {changed}/20")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Avg tokens/q:      {total_tokens//20:,}")

    print("\n  Per-question comparison:")
    print(f"  {'qid':<12} {'p2v8':>6} {'gold':>6} {'p2v7':>6} {'sc':>4} {'match':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*6}")
    for r in results:
        match_str = "OK" if r["matches_gold"] else "MISS"
        print(
            f"  {r['qid']:<12} {r['answer']:>6} {r['pseudo_gold']:>6} "
            f"{r['p2v7_answer']:>6} "
            f"{'Y' if r['self_checked'] else 'N':>4} "
            f"{match_str:>6}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
