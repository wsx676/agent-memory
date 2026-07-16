from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# #region debug-point A:report-helper
def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict[str, object]) -> None:
    env_path = ROOT / ".dbg" / "pdf-preprocess-gap.env"
    debug_server_url = "http://127.0.0.1:7777/event"
    debug_session_id = "pdf-preprocess-gap"
    try:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    debug_session_id = line.split("=", 1)[1].strip()
        payload = {
            "sessionId": debug_session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                debug_server_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass


# #endregion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="要处理的原始文件绝对路径")
    parser.add_argument("--output", required=True, help="单文件处理结果输出 JSON 文件绝对路径")
    args = parser.parse_args()

    from scripts.preprocess_public_dataset_a import process_file

    file_path = Path(args.file)
    output_path = Path(args.output)
    # #region debug-point A:single-start
    _debug_report(
        "A",
        "process_single_public_dataset_a.py:main:start",
        "[DEBUG] single file preprocess start",
        {"file": str(file_path), "output": str(output_path)},
    )
    # #endregion
    processed_doc, chunk_dicts = process_file(file_path)
    # #region debug-point A:single-processed
    _debug_report(
        "A",
        "process_single_public_dataset_a.py:main:processed",
        "[DEBUG] single file preprocess returned",
        {
            "file": str(file_path),
            "doc_id": processed_doc.doc_id,
            "status": processed_doc.status,
            "chunk_count": len(chunk_dicts),
        },
    )
    # #endregion
    output_path.write_text(
        json.dumps(
            {"processed_doc": asdict(processed_doc), "chunks": chunk_dicts},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # #region debug-point A:single-output
    _debug_report(
        "A",
        "process_single_public_dataset_a.py:main:output",
        "[DEBUG] single file output written",
        {"file": str(file_path), "output_exists": output_path.exists(), "output": str(output_path)},
    )
    # #endregion


if __name__ == "__main__":
    main()
