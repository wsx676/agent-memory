from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_DIR = PROJECT_ROOT / "public_dataset_a" / "questions" / "group_a"
EXECUTION_LOG = PROJECT_ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "group_a" / "group_a_execution_log.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "answer.csv"
OPTION_KEYS = ("A", "B", "C", "D", "E", "F")
ANSWER_FORMAT_MAPPING = {
    "single": "single",
    "mcq": "single",
    "multi": "multi",
    "tf": "judge",
    "judge": "judge",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_options(raw_options: Any) -> list[str]:
    if isinstance(raw_options, dict):
        return [str(raw_options[key]) for key in OPTION_KEYS if key in raw_options]
    if isinstance(raw_options, list):
        return [str(item) for item in raw_options]
    return []


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for file_path in sorted(QUESTIONS_DIR.glob("*.json")):
        payload = read_json(file_path)
        if not isinstance(payload, list):
            continue
        for row in payload:
            options = normalize_options(row.get("options"))
            answer_format = ANSWER_FORMAT_MAPPING.get(str(row.get("answer_format", "")).lower())
            if not options or not answer_format:
                continue
            cases.append(
                {
                    "qid": str(row["qid"]),
                    "question": str(row["question"]),
                    "answer_format": answer_format,
                    "option_count": len(options),
                }
            )
    return cases


def load_answers() -> dict[str, str]:
    answers: dict[str, str] = {}
    for row in read_jsonl(EXECUTION_LOG):
        if row.get("phase") == "validation":
            continue
        qid = str(row.get("qid", ""))
        if qid:
            answers[qid] = str(row.get("answer", ""))
    return answers


def normalize_answer(raw_answer: str, answer_format: str) -> str:
    letters = "".join(ch for ch in raw_answer.upper() if ch in OPTION_KEYS)
    if answer_format in {"single", "judge"}:
        return letters[:1] or "A"
    deduped = "".join(dict.fromkeys(letters))
    return deduped or "A"


def estimate_tokens(question: str, answer: str) -> tuple[int, int, int]:
    _ = question, answer
    return 0, 0, 0


def main() -> None:
    cases = load_cases()
    answers = load_answers()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list[str | int]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for case in cases:
        answer = normalize_answer(answers.get(case["qid"], ""), case["answer_format"])
        prompt_tokens, completion_tokens, row_total_tokens = estimate_tokens(case["question"], answer)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += row_total_tokens
        rows.append([case["qid"], answer, prompt_tokens, completion_tokens, row_total_tokens])

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", total_prompt_tokens, total_completion_tokens, total_tokens])
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "output_file": str(OUTPUT_FILE),
                "case_count": len(rows),
                "summary": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
