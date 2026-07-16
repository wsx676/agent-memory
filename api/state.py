from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_FILE = DATA_DIR / "documents.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
TASKS_FILE = DATA_DIR / "tasks.json"
RESULTS_FILE = DATA_DIR / "results.json"
EVIDENCE_FILE = DATA_DIR / "evidence.json"
ANSWER_FILE = DATA_DIR / "answer.csv"


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for file_path, default_data in (
        (DOCUMENTS_FILE, []),
        (CHUNKS_FILE, []),
        (TASKS_FILE, []),
        (RESULTS_FILE, []),
        (EVIDENCE_FILE, []),
    ):
        if not file_path.exists():
            file_path.write_text(json.dumps(default_data, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ANSWER_FILE.exists():
        ANSWER_FILE.write_text("qid,answer,prompt_tokens,completion_tokens,total_tokens\n", encoding="utf-8")


def read_json(path: Path) -> list[dict[str, Any]]:
    ensure_data_files()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    ensure_data_files()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
