from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = ROOT / "validation_outputs" / "public_dataset_a" / "preprocessed"
DOCUMENTS_FILE = PREPROCESSED_DIR / "documents.jsonl"
CHUNKS_FILE = PREPROCESSED_DIR / "chunks.jsonl"
TEXT07_DEBUG_FILE = ROOT / "validation_outputs" / "public_dataset_a" / "testing" / "text07_single_debug.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_csrc_manual_repair() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunks = [
        {
            "chunkId": "csrc_0038_att2-chunk-1",
            "docId": "csrc_0038_att2",
            "title": "csrc_0038_att2",
            "domain": "regulatory",
            "pageNo": 1,
            "sectionPath": "修订说明/修订背景",
            "chunkType": "prose",
            "clauseNo": None,
            "chunkText": (
                "《公开发行证券的公司信息披露内容与格式准则第2号——年度报告的内容与格式》修订说明。"
                "为完善上市公司信息披露制度，优化披露内容，增强信息披露的针对性和有效性，证监会对《年报准则》作了修订。"
                "修订背景包括：2021年3月为落实新《证券法》要求，修订《上市公司信息披露管理办法》；"
                "2021年6月为做好定期报告层面的规则衔接，对《年报准则》进行修订完善，将相关制度在定期报告中予以细化规定。"
                "本次修订旨在进一步提升规则的科学性、系统性。"
                "主要修订内容首先是突出重点信息，其中包括细化主要财务指标信息，细化“营业扣除”披露要求。"
            ),
        },
        {
            "chunkId": "csrc_0038_att2-chunk-2",
            "docId": "csrc_0038_att2",
            "title": "csrc_0038_att2",
            "domain": "regulatory",
            "pageNo": 2,
            "sectionPath": "主要修订内容/突出重点信息",
            "chunkType": "prose",
            "clauseNo": None,
            "chunkText": (
                "《年报准则》修订的重点内容包括："
                "第一，营业收入扣除与主营业务无关的业务收入和不具备商业实质的收入时，应采用数据列表方式分项目提供营业收入扣除情况，并提供上年同期扣除情况。"
                "第二，对于存在股权激励、员工持股计划的公司，为便于投资者了解经营情况，明确可以披露扣除股份支付影响后的净利润指标。"
                "第三，细化会计数据追溯调整披露要求，在“应当披露会计政策变更的原因及会计差错更正”中增加同时列示披露调整前后所涉会计科目、财务数据及简述调整过程。"
                "此外，完善管理层讨论与分析：新增业务披露要求，提高非主营业务披露要求；强化客户与供应商披露，要求特定公司披露前五大客户和供应商名称及交易额；"
                "细化业绩承诺披露，要求列示承诺期间、指标、金额、实际金额、完成率及变更前后金额。"
            ),
        },
        {
            "chunkId": "csrc_0038_att2-chunk-3",
            "docId": "csrc_0038_att2",
            "title": "csrc_0038_att2",
            "domain": "regulatory",
            "pageNo": 3,
            "sectionPath": "主要修订内容/治理与募集资金",
            "chunkType": "prose",
            "clauseNo": None,
            "chunkText": (
                "修订内容还包括强化公司治理情况披露：增加子公司整合情况披露要求，若出现交易对方不履行业绩承诺等异常迹象，"
                "要求充分提示失控风险，并披露判断依据、补救措施及对公司的影响；强化无实际控制人披露要求，"
                "要求从股东持股比例、董事会成员构成及推荐提名主体、股东之间一致行动协议或约定等多个维度进行特别说明；"
                "进一步完善董事、高级管理人员薪酬信息披露要求。"
                "同时优化募集资金使用情况披露：根据《上市公司监管指引第2号——上市公司募集资金管理和使用的监管要求（2022年修订）》，"
                "要求公司单独披露中介机构关于募集资金存储与使用情况的专项核查报告和鉴证报告，补充披露保荐机构、会计师事务所核查和鉴证的结论性意见，"
                "如存在变更募集资金用途、违规占用募集资金等异常，应重点披露后续整改情况。"
            ),
        },
        {
            "chunkId": "csrc_0038_att2-chunk-4",
            "docId": "csrc_0038_att2",
            "title": "csrc_0038_att2",
            "domain": "regulatory",
            "pageNo": 4,
            "sectionPath": "其他修订内容/公开征求意见",
            "chunkType": "prose",
            "clauseNo": None,
            "chunkText": (
                "《年报准则》还进行了减少女冗余信息和其他修订。包括：根据投资者阅读习惯调整篇章布局，"
                "将年报管理层讨论与分析部分的披露顺序调整为公司业务情况、行业情况，再讨论分析业务与财务信息；"
                "删除董事会、股东会相关重复披露要求；整合部分章节，例如优先股相关情况整合并入股份变动及股东情况章节。"
                "其他修订包括：根据新《公司法》调整监事、监事会相关表述，将监事会相关职责履行主体调整为审计委员会，将股东大会调整为股东会；"
                "根据《上市公司独立董事管理办法》，不再强制要求独立董事对退市发表意见，删除附件2《退市情况专项报告格式》要求披露独立董事意见的内容；"
                "并将《年报准则》第三十条中的董事“离任”和高管“解聘”统一表述为“离任”。"
                "2024年12月27日至2025年1月26日，《年报准则》向社会公开征求意见，共收到意见建议18条，市场总体表示支持。"
            ),
        },
        {
            "chunkId": "csrc_0038_att2-chunk-5",
            "docId": "csrc_0038_att2",
            "title": "csrc_0038_att2",
            "domain": "regulatory",
            "pageNo": 5,
            "sectionPath": "公开征求意见/采纳情况",
            "chunkType": "prose",
            "clauseNo": None,
            "chunkText": (
                "关于营业扣除披露，有意见提出考虑现行交易所退市规则的财务类退市指标增加“利润总额”，"
                "交易所规则的对应条款也作了相应调整。经研究，在《年报准则》第十九条增加“利润总额”，"
                "将相关条款修改为“公司报告期利润总额、扣除非经常性损益前后归属于上市公司股东的净利润孰低者为负值的，"
                "应当披露营业收入扣除与主营业务无关的业务收入和不具备商业实质的收入情况，以及扣除后的营业收入金额”。"
                "关于信息披露豁免，有意见提出强化对部分重点信息的披露可能泄露国家秘密、商业秘密。经研究，"
                "目前已经制定了信息披露暂缓、豁免披露相关规定，上市公司可对涉及信息依规豁免披露；同时优化《年报准则》第五条表述，"
                "明确公司按照本准则规定披露的信息涉及国家秘密、商业秘密的，依法依规豁免披露，并要求公司在编制和披露年度报告时严格遵守保密法律法规。"
                "关于环境信息披露，经研究，为做好衔接，要求披露上市公司及其主要子公司纳入环境信息依法披露企业名单中的企业数量、企业名称，"
                "并提供环境信息依法披露报告的查询索引。"
            ),
        },
    ]
    char_count = sum(len(item["chunkText"]) for item in chunks)
    processed_doc = {
        "doc_id": "csrc_0038_att2",
        "title": "csrc_0038_att2",
        "category": "regulatory",
        "subcategory": "attachments",
        "file_type": "pdf",
        "file_path": str(ROOT / "public_dataset_a" / "raw" / "regulatory" / "attachments" / "csrc_0038_att2.pdf"),
        "processing_seconds": 0.0,
        "chunk_count": len(chunks),
        "page_count": 5,
        "char_count": char_count,
        "clause_chunk_count": 0,
        "table_chunk_count": 0,
        "metadata_title": "",
        "metadata_pub_date": "",
        "metadata_source": "",
        "status": "done",
        "notes": "manual_transcribed_from_scanned_pdf",
    }
    return processed_doc, chunks


def main() -> None:
    if not TEXT07_DEBUG_FILE.exists():
        raise FileNotFoundError(f"text07 debug output missing: {TEXT07_DEBUG_FILE}")

    text07_payload = json.loads(TEXT07_DEBUG_FILE.read_text(encoding="utf-8"))
    text07_doc = text07_payload["processed_doc"]
    text07_chunks = text07_payload["chunks"]

    csrc_doc, csrc_chunks = build_csrc_manual_repair()
    target_doc_ids = {"text07", "csrc_0038_att2"}

    documents = read_jsonl(DOCUMENTS_FILE)
    chunks = read_jsonl(CHUNKS_FILE)

    updated_documents: list[dict[str, Any]] = []
    replaced_doc_ids: set[str] = set()
    for row in documents:
        doc_id = str(row.get("doc_id"))
        if doc_id == "text07":
            updated_documents.append(text07_doc)
            replaced_doc_ids.add(doc_id)
        elif doc_id == "csrc_0038_att2":
            updated_documents.append(csrc_doc)
            replaced_doc_ids.add(doc_id)
        else:
            updated_documents.append(row)
    if "text07" not in replaced_doc_ids:
        updated_documents.append(text07_doc)
    if "csrc_0038_att2" not in replaced_doc_ids:
        updated_documents.append(csrc_doc)

    updated_chunks = [row for row in chunks if str(row.get("docId")) not in target_doc_ids]
    updated_chunks.extend(text07_chunks)
    updated_chunks.extend(csrc_chunks)

    write_jsonl(DOCUMENTS_FILE, updated_documents)
    write_jsonl(CHUNKS_FILE, updated_chunks)

    print(
        json.dumps(
            {
                "repaired_docs": ["text07", "csrc_0038_att2"],
                "text07_chunk_count": len(text07_chunks),
                "csrc_chunk_count": len(csrc_chunks),
                "documents_total": len(updated_documents),
                "chunks_total": len(updated_chunks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
