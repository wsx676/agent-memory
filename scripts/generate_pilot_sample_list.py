from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "public_dataset_a" / "raw"
OUTPUT_ROOT = ROOT / "validation_outputs" / "public_dataset_a"
OUTPUT_CSV = OUTPUT_ROOT / "pilot_sample_list_10pct.csv"
OUTPUT_JSON = OUTPUT_ROOT / "pilot_sample_list_10pct_summary.json"
RANDOM_SEED = 20260625


@dataclass
class SampleItem:
    sample_id: str
    stratum: str
    category: str
    subcategory: str
    file_name: str
    doc_id: str
    extension: str
    file_size_mb: float
    file_path: str
    validation_focus: str


STRATA_CONFIG = [
    {
        "stratum": "financial_contracts_pdf",
        "path": DATASET_ROOT / "financial_contracts",
        "glob": "*.pdf",
        "sample_size": 2,
        "validation_focus": "条款编号、责任边界、跨段条件、表格/附表一致性",
    },
    {
        "stratum": "financial_reports_pdf",
        "path": DATASET_ROOT / "financial_reports",
        "glob": "*.pdf",
        "sample_size": 1,
        "validation_focus": "目录噪声、财务表格还原、关键指标与页码映射",
    },
    {
        "stratum": "insurance_pdf",
        "path": DATASET_ROOT / "insurance",
        "glob": "*.pdf",
        "sample_size": 2,
        "validation_focus": "免责条款、给付规则、等待期、数值与条件抽取",
    },
    {
        "stratum": "research_pdf",
        "path": DATASET_ROOT / "research",
        "glob": "*.pdf",
        "sample_size": 2,
        "validation_focus": "长段落抽取、观点块切分、风险提示与正文噪声区分",
    },
    {
        "stratum": "regulatory_html",
        "path": DATASET_ROOT / "regulatory" / "html",
        "glob": "*.html",
        "sample_size": 38,
        "validation_focus": "网页正文清洗、标题/发布日期抽取、导航噪声过滤",
    },
    {
        "stratum": "regulatory_attachments_pdf",
        "path": DATASET_ROOT / "regulatory" / "attachments",
        "glob": "*.pdf",
        "sample_size": 13,
        "validation_focus": "网页附件映射、附件正文抽取、页码与条款回指",
    },
    {
        "stratum": "regulatory_txt",
        "path": DATASET_ROOT / "regulatory" / "txt",
        "glob": "*.txt",
        "sample_size": 1,
        "validation_focus": "章节结构、目录去重、条款边界与规范化文本保真",
    },
]


def detect_doc_id(path: Path) -> str:
    return path.stem


def build_sample_item(index: int, stratum: str, file_path: Path, validation_focus: str) -> SampleItem:
    relative_parts = file_path.relative_to(DATASET_ROOT).parts
    category = relative_parts[0]
    subcategory = relative_parts[1] if len(relative_parts) > 2 else "root"
    return SampleItem(
        sample_id=f"S{index:03d}",
        stratum=stratum,
        category=category,
        subcategory=subcategory,
        file_name=file_path.name,
        doc_id=detect_doc_id(file_path),
        extension=file_path.suffix.lower(),
        file_size_mb=round(file_path.stat().st_size / 1024 / 1024, 3),
        file_path=str(file_path),
        validation_focus=validation_focus,
    )


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    samples: list[SampleItem] = []
    summary: dict[str, dict[str, object]] = {}

    for config in STRATA_CONFIG:
        all_files = sorted(config["path"].glob(config["glob"]))
        chosen = rng.sample(all_files, config["sample_size"])
        chosen = sorted(chosen)
        summary[config["stratum"]] = {
            "population": len(all_files),
            "sample_size": config["sample_size"],
            "sampling_ratio": round(config["sample_size"] / len(all_files), 4),
        }
        for file_path in chosen:
            samples.append(
                build_sample_item(
                    index=len(samples) + 1,
                    stratum=config["stratum"],
                    file_path=file_path,
                    validation_focus=config["validation_focus"],
                )
            )

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "total_samples": len(samples),
                "summary": summary,
                "output_csv": str(OUTPUT_CSV),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "total_samples": len(samples),
                "output_csv": str(OUTPUT_CSV),
                "output_json": str(OUTPUT_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
