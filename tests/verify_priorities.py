"""验证两个优先级优化的实际行为。

优先级1: financial_reports 表格恢复（scripts/preprocess_public_dataset_a.py 的开关）
优先级2: 大表格按行切分 + 表头重复（api/services/preprocess.py 的 _split_table_part）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.services import preprocess as P  # noqa: E402


def build_table(rows: int) -> str:
    """构造 [表格] 标记 + N 行两列表格，首行为列头。"""
    lines = ["[表格]", "项目 | 本期金额 | 上期金额"]
    for i in range(1, rows + 1):
        lines.append(f"科目{i} | {i*100} | {i*90}")
    return "\n".join(lines)


def test_split_table_part_header_repeat():
    print("=== 优先级2: _split_table_part 表头重复验证 ===")
    # 一个 60 行的长表，按长度 SPLIT_TARGET_LENGTH=1000 应被切成多块
    part = build_table(60)
    blocks = P._split_table_part(part)
    print(f"表格总行数=60, 切成 {len(blocks)} 块")
    for idx, b in enumerate(blocks, 1):
        first_real_line = b.splitlines()[1] if len(b.splitlines()) > 1 else "<无>"
        nl = len(b.splitlines()) - 1  # 去掉 [表格] 标记行
        print(f"  块{idx}: 行数={nl}, 第二行(应为列头)={first_real_line!r}")
    # 关键断言：除第一块外，每块都应保留列头行 "项目 | 本期金额 | 上期金额"
    header = "项目 | 本期金额 | 上期金额"
    ok = all(b.split("\n", 1)[1].split("\n", 1)[0] == header for b in blocks[1:])
    print(f"  [断言] 后续块均重复列头行: {ok}")
    return ok


def test_split_table_part_no_truncation():
    print("\n=== 优先级2: 不再 [:1200] 硬截断验证 ===")
    part = build_table(80)  # 长表
    blocks = P._split_table_part(part)
    total_rows = sum(len(b.splitlines()) - 1 for b in blocks)
    print(f"原始 80 行 -> 切 {len(blocks)} 块, 合计保留 {total_rows} 行 (含各块重复列头)")
    # 内容无丢失：所有科目行都应出现在某块中
    all_text = "\n".join(blocks)
    missing = [i for i in range(1, 81) if f"科目{i} " not in all_text]
    print(f"  [断言] 无科目行丢失: {not missing} (缺失={missing[:5]})")
    return not missing


def test_e2e_financial_report():
    print("\n=== 优先级1: 年报 PDF 端到端表格恢复验证 ===")
    from api.models import DocumentRecord

    pdf_dir = ROOT / "public_dataset_a" / "raw" / "financial_reports"
    # 选一个中小体积年报做快速验证（<8MB，原开关也会放行）
    target = pdf_dir / "annual_catl_2024_report.PDF"
    if not target.exists():
        print("未找到目标年报文件，跳过端到端验证")
        return None
    record = DocumentRecord(
        documentId=target.stem,
        title=target.stem,
        fileType="pdf",
        sourcePath=str(target),
        status="queued",
        chunkCount=0,
    )
    # 模拟 preprocess_public_dataset_a.py 对 financial_reports 的开关
    size = target.stat().st_size
    enable_table_recovery = (
        record.file_type == "pdf"
        and size <= 64 * 1024 * 1024
        and not False  # not prefer_pdfplumber for financial_reports
        and not (False)  # not regulatory/attachments
    )
    print(f"enable_table_recovery={enable_table_recovery} (size={size/1024/1024:.2f}MB)")
    chunks = P.preprocess_document(record, domain="financial_reports", enable_ocr=False, enable_table_recovery=enable_table_recovery)
    tables = [c for c in chunks if c.chunk_type == "table"]
    print(f"总切片={len(chunks)}, 表格切片={len(tables)}")
    sample = tables[0].chunk_text if tables else ""
    print(f"  首个表格切片前80字: {sample[:80]!r}")
    return len(tables)


if __name__ == "__main__":
    r2a = test_split_table_part_header_repeat()
    r2b = test_split_table_part_no_truncation()
    r1 = test_e2e_financial_report()
    print("\n=== 汇总 ===")
    print(f"优先级2 表头重复: {'PASS' if r2a else 'FAIL(缺口)'}")
    print(f"优先级2 无截断丢失: {'PASS' if r2b else 'FAIL'}")
    print(f"优先级1 年报表格恢复: {'PASS' if (r1 is not None and r1 > 0) else 'FAIL/需复核'} (tables={r1})")
