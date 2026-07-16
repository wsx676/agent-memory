from __future__ import annotations

import html
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from api.models import DocumentRecord, StructuredChunk


OCR_ENGINE: Any | None = None
OCR_INIT_ATTEMPTED = False


@dataclass
class PageContent:
    page_no: int
    text: str
    section_path: str
    chunk_type: str = "prose"
    is_toc: bool = False


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_clause_no(text: str) -> str | None:
    matched = re.match(r"\s*(第[\d一二三四五六七八九十百千万]+条)", text)
    return matched.group(1) if matched else None


def _detect_section_path(text: str, page_no: int) -> str:
    heading_patterns = [
        r"(第[\d一二三四五六七八九十百千万]+章[^\n]{0,30})",
        r"(第[\d一二三四五六七八九十百千万]+节[^\n]{0,30})",
        r"([一二三四五六七八九十]+、[^\n]{0,30})",
    ]
    for pattern in heading_patterns:
        matched = re.search(pattern, text)
        if matched:
            return matched.group(1).strip()
    return f"第{page_no}页"


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _split_lines(text: str) -> list[str]:
    return [_normalize_line(line) for line in text.splitlines() if _normalize_line(line)]


def _looks_like_page_marker(line: str) -> bool:
    normalized = _normalize_line(line)
    if not normalized:
        return True
    return bool(
        re.fullmatch(r"(第?\s*\d+\s*页|page\s*\d+|\d+/\d+|\d+)", normalized, re.IGNORECASE)
    )


def _is_noise_line(line: str) -> bool:
    normalized = _normalize_line(line)
    if not normalized:
        return True
    if _looks_like_page_marker(normalized):
        return True
    noise_patterns = [
        r"仅供参考",
        r"请仔细阅读",
        r"版权所有",
        r"联系电话[:：]?",
        r"网址[:：]?",
        r"www\.",
        r"https?://",
    ]
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in noise_patterns)


def _looks_like_toc(text: str) -> bool:
    lines = _split_lines(text)
    if not lines:
        return False
    joined = "\n".join(lines[:15])
    if re.search(r"目录|contents?", joined, re.IGNORECASE):
        return True
    toc_line_count = 0
    for line in lines[:20]:
        if re.search(r"(\.{2,}|·{2,}|…{2,}).{0,10}\d+$", line) or re.search(r"\b\d+\s*$", line):
            toc_line_count += 1
    return toc_line_count >= 5


def _find_repeated_edge_lines(page_texts: list[str]) -> tuple[set[str], set[str]]:
    header_counts: dict[str, int] = {}
    footer_counts: dict[str, int] = {}
    for text in page_texts:
        lines = _split_lines(text)
        if not lines:
            continue
        first_line = lines[0]
        last_line = lines[-1]
        if 3 <= len(first_line) <= 80 and not _looks_like_page_marker(first_line):
            header_counts[first_line] = header_counts.get(first_line, 0) + 1
        if 1 <= len(last_line) <= 80:
            footer_counts[last_line] = footer_counts.get(last_line, 0) + 1

    min_repeat = 2 if len(page_texts) <= 3 else 3
    headers = {line for line, count in header_counts.items() if count >= min_repeat}
    footers = {line for line, count in footer_counts.items() if count >= min_repeat}
    return headers, footers


def _clean_page_text(text: str, repeated_headers: set[str], repeated_footers: set[str]) -> str:
    lines = _split_lines(text)
    cleaned_lines: list[str] = []
    previous_line = ""
    for index, line in enumerate(lines):
        if index == 0 and line in repeated_headers:
            continue
        if index == len(lines) - 1 and line in repeated_footers:
            continue
        if _is_noise_line(line):
            continue
        if line == previous_line:
            continue
        cleaned_lines.append(line)
        previous_line = line
    return _normalize_text("\n".join(cleaned_lines))


def _postprocess_pages(pages: list[PageContent]) -> list[PageContent]:
    if not pages:
        return []
    repeated_headers, repeated_footers = _find_repeated_edge_lines([page.text for page in pages])
    processed_pages: list[PageContent] = []
    seen_full_pages: set[str] = set()
    for page in pages:
        cleaned_text = _clean_page_text(page.text, repeated_headers, repeated_footers)
        if not cleaned_text:
            continue
        is_toc = _looks_like_toc(cleaned_text)
        # 仅对完全相同的页面去重（整页归一化文本做键），避免前缀相同的
        # 合同/条款/报表页被误删。真正的重复页（扫描重复、模板复用）仍会被去除。
        dedupe_key = cleaned_text
        if dedupe_key in seen_full_pages:
            continue
        seen_full_pages.add(dedupe_key)
        if is_toc:
            continue
        processed_pages.append(
            PageContent(
                page_no=page.page_no,
                text=cleaned_text,
                section_path=_detect_section_path(cleaned_text, page.page_no),
                chunk_type=page.chunk_type,
                is_toc=is_toc,
            )
        )
    return processed_pages


def _format_timestamp(raw_ts: str) -> str:
    """将 Unix 时间戳（秒或毫秒）格式化为 YYYY-MM-DD 字符串。

    原 pub_date 提取直接把时间戳前 10 位当日期用（如 1714475760），
    既不可读也不是标准日期格式。
    """
    try:
        ts = int(raw_ts)
        if ts > 10_000_000_000:  # 毫秒级
            ts //= 1000
        import time

        return time.strftime("%Y-%m-%d", time.localtime(ts))
    except (ValueError, OSError):
        return raw_ts[:10]


def _extract_html_meta(raw_html: str, name: str) -> str:
    patterns = [
        rf'<meta[^>]+name=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
        rf'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']{re.escape(name)}["\']',
    ]
    for pattern in patterns:
        matched = re.search(pattern, raw_html, re.IGNORECASE | re.DOTALL)
        if matched:
            return _normalize_text(matched.group(1))
    return ""


def _extract_html_text(raw_html: str, pattern: str) -> str:
    matched = re.search(pattern, raw_html, re.IGNORECASE | re.DOTALL)
    if not matched:
        return ""
    return _normalize_text(_strip_html_tags(matched.group(1)))


def extract_html_metadata(file_path: Path) -> dict[str, str]:
    raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
    title = _extract_html_meta(raw_html, "ArticleTitle")
    if not title:
        title_match = re.search(r"<title>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
        title = _normalize_text(title_match.group(1)) if title_match else ""
    pub_date = _extract_html_meta(raw_html, "PubDate")
    if not pub_date:
        timestamp_match = re.search(r'class=["\']fwrq["\'][^>]+value=[\'"](\d{10,13})[\'"]', raw_html, re.IGNORECASE)
        if timestamp_match:
            pub_date = _format_timestamp(timestamp_match.group(1))
        else:
            pub_date = _extract_html_text(raw_html, r"<th>\s*发文日期\s*</th>\s*<td[^>]*>(.*?)</td>")
    source = _extract_html_meta(raw_html, "ContentSource")
    if not source:
        source = _extract_html_meta(raw_html, "SiteName")
    if not source:
        source = _extract_html_text(raw_html, r"<th>\s*发布机构\s*</th>\s*<td[^>]*>(.*?)</td>")
    if not source:
        footer_source_match = re.search(r"主办单位[:：]\s*([^<\n]+)", raw_html, re.IGNORECASE)
        if footer_source_match:
            source = _normalize_text(footer_source_match.group(1))
    return {
        "title": title,
        "pub_date": pub_date,
        "source": source,
    }


def _strip_html_tags(html_text: str) -> str:
    # 剔除脚本/样式/表单等非正文块
    text = re.sub(r"(?is)<(script|style|noscript|form|svg|footer|header|nav|aside).*?>.*?</\1>", " ", html_text)
    # 剔除带导航/菜单/面包屑 class 的 div 块（政府网站常见噪声源）
    text = re.sub(
        r'(?is)<div[^>]+class=["\'][^"\']*(?:nav|menu|breadcrumb|sidebar|top-bar|head-box|footer|header|link|share|print|close)[^"\']*["\'][^>]*>.*?</div>',
        " ",
        text,
    )
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|li|tr|section|article|h\d)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return _normalize_text(text)


def _extract_balanced_block(html_body: str, tag: str, start: int) -> str | None:
    """从 start 位置的 <tag ...> 开始，按标签深度匹配提取到对应的闭合标签。

    解决正则 (.*?) 在嵌套 div 时遇到第一个 </div> 就提前关闭的问题。
    返回标签内部内容（不含外层标签）；若深度不匹配则返回 None。
    """
    open_re = re.compile(rf"(?is)<{tag}[^>]*>", )
    close_re = re.compile(rf"(?is)</{tag}>")
    # 确认 start 处确实是一个开标签
    open_match = open_re.match(html_body, start)
    if not open_match:
        return None
    pos = open_match.end()
    depth = 1
    while depth > 0 and pos < len(html_body):
        next_open = open_re.search(html_body, pos)
        next_close = close_re.search(html_body, pos)
        if not next_close:
            return None
        if next_open and next_open.start() < next_close.start():
            depth += 1
            pos = next_open.end()
        else:
            depth -= 1
            if depth == 0:
                return html_body[open_match.end():next_close.start()]
            pos = next_close.end()
    return None


# 正文容器的 class/id 关键词，按优先级排序
_CONTENT_KEYWORDS = (
    "detail-news", "article-content", "TRS_Editor", "Custom_UnionStyle",
    "zoom", "articleContent", "article_body",
)
_CONTENT_FALLBACK_KEYWORDS = (
    "content", "article", "main", "detail", "正文", "news",
)


def _select_html_content(raw_html: str) -> str:
    body_match = re.search(r"(?is)<body[^>]*>(.*?)</body>", raw_html)
    html_body = body_match.group(1) if body_match else raw_html

    # 在 body 中查找所有 <div>/<article>/<section> 开标签的位置
    tag_open_re = re.compile(r"(?is)<(div|article|section)[^>]*>")

    def collect_candidates(keywords: tuple[str, ...]) -> list[str]:
        candidates: list[str] = []
        for match in tag_open_re.finditer(html_body):
            tag = match.group(1)
            attrs = match.group(0).lower()
            if not any(kw.lower() in attrs for kw in keywords):
                continue
            inner = _extract_balanced_block(html_body, tag, match.start())
            if inner is None:
                continue
            cleaned = _strip_html_tags(inner)
            if len(cleaned) > 40:
                candidates.append(cleaned)
        return candidates

    # 第一轮：高置信度正文容器
    preferred = collect_candidates(_CONTENT_KEYWORDS)
    if preferred:
        return max(preferred, key=len)

    # 第二轮：通用正文容器关键词
    fallback = collect_candidates(_CONTENT_FALLBACK_KEYWORDS)
    if fallback:
        # 取最长的候选，但要求至少 200 字以排除导航栏
        long_fallback = [c for c in fallback if len(c) >= 200]
        if long_fallback:
            return max(long_fallback, key=len)

    # 最后回退：清洗整个 body
    return _strip_html_tags(html_body)


def _read_html_pages(file_path: Path) -> list[PageContent]:
    raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
    metadata = extract_html_metadata(file_path)
    article_text = _select_html_content(raw_html)
    header_parts = [metadata["title"], metadata["pub_date"], metadata["source"]]
    page_text = "\n".join([part for part in header_parts if part])
    if article_text:
        page_text = "\n\n".join([page_text, article_text]) if page_text else article_text
    return _postprocess_pages(
        [PageContent(page_no=1, text=page_text, section_path=_detect_section_path(page_text, 1), chunk_type="prose")]
    )


def _serialize_table(table: list[list[str | None]]) -> str:
    rows: list[str] = []
    for row in table:
        cells = [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_tables_from_pdf(file_path: Path) -> dict[int, list[str]]:
    try:
        import pdfplumber  # type: ignore

        page_tables: dict[int, list[str]] = {}
        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                serialized_tables = []
                for table in page.extract_tables() or []:
                    serialized = _serialize_table(table)
                    if serialized:
                        serialized_tables.append(serialized)
                if serialized_tables:
                    page_tables[page_index] = serialized_tables
        return page_tables
    except Exception:
        return {}


def _read_pdf_pages_with_pdfplumber(file_path: Path, *, enable_table_recovery: bool) -> list[PageContent]:
    try:
        import pdfplumber  # type: ignore

        pages: list[PageContent] = []
        with pdfplumber.open(file_path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                page_text = _normalize_text(page.extract_text() or "")
                table_texts: list[str] = []
                if enable_table_recovery:
                    for table in page.extract_tables() or []:
                        serialized = _serialize_table(table)
                        if serialized:
                            table_texts.append(serialized)
                if table_texts:
                    parts = [page_text] if page_text else []
                    parts.extend(f"[表格]\n{table}" for table in table_texts)
                    page_text = "\n\n".join(parts)
                pages.append(
                    PageContent(
                        page_no=page_index,
                        text=page_text,
                        section_path=_detect_section_path(page_text, page_index),
                        chunk_type="prose",
                    )
                )
        return _postprocess_pages(pages)
    except Exception:
        return []


def _get_ocr_engine() -> Any | None:
    """初始化 OCR 引擎。

    优先使用 RapidOCR（ONNX 推理，无 paddlepaddle 依赖，兼容性好）；
    回退到 PaddleOCR。引擎实例缓存在全局变量中避免重复初始化。
    """
    global OCR_ENGINE, OCR_INIT_ATTEMPTED
    if OCR_ENGINE is not None:
        return OCR_ENGINE
    if OCR_INIT_ATTEMPTED:
        return None
    OCR_INIT_ATTEMPTED = True
    # 首选 RapidOCR
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        OCR_ENGINE = RapidOCR()
        return OCR_ENGINE
    except Exception:
        pass
    # 回退 PaddleOCR
    try:
        from paddleocr import PaddleOCR  # type: ignore

        try:
            OCR_ENGINE = PaddleOCR(use_textline_orientation=True, lang="ch")
        except TypeError:
            OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        return OCR_ENGINE
    except Exception:
        OCR_ENGINE = None
        return None


def _extract_ocr_text(page: Any) -> str:
    engine = _get_ocr_engine()
    if engine is None:
        return ""
    try:
        import fitz  # type: ignore

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    except Exception:
        pix = page.get_pixmap()
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        img_array = np.array(image)
        engine_type = type(engine).__module__
        if "rapidocr" in engine_type:
            # RapidOCR: 返回 (result, elapsed)
            result, _ = engine(img_array)
            if result:
                return "\n".join(str(item[1]) for item in result if item and len(item) > 1)
            return ""
        # PaddleOCR: 返回嵌套列表
        try:
            result = engine.ocr(img_array) or []
        except TypeError:
            result = engine.ocr(img_array, cls=True) or []
        lines: list[str] = []
        for item in result:
            for line in item or []:
                if len(line) > 1 and line[1]:
                    lines.append(str(line[1][0]))
        return "\n".join(lines)
    except Exception:
        return ""


def _needs_ocr(page_text: str, page: Any) -> bool:
    """判断页面是否需要 OCR 补充。

    扫描版页面通常提取出的文本极少，但包含整页图像。仅用固定字数阈值会漏掉
    提取出少量噪声文字（页眉/水印）的扫描页。这里综合两个信号判断：
    - 文本极短（少于 120 字），且
    - 页面图像面积占比高（存在覆盖页面较大部分的图片块），说明正文大概率是图片。
    """
    if len(page_text) >= 120:
        return False
    try:
        images = page.get_images(full=True) or []
        if not images:
            # 无嵌入图片但文本也极少：疑似空白页或解析失败，仍尝试 OCR
            return len(page_text) < 40
        page_area = float(page.rect.width) * float(page.rect.height)
        image_area = 0.0
        for info in images:
            xref = info[0]
            for rect in page.get_image_rects(xref) or []:
                image_area += float(rect.width) * float(rect.height)
        # 图像覆盖页面超过 30% 且文本极少，按扫描页处理
        return page_area > 0 and image_area / page_area >= 0.30
    except Exception:
        return len(page_text) < 40


def _read_pdf_pages(file_path: Path, *, enable_ocr: bool, enable_table_recovery: bool, prefer_pdfplumber: bool = False) -> list[PageContent]:
    if prefer_pdfplumber:
        return _read_pdf_pages_with_pdfplumber(
            file_path,
            enable_table_recovery=enable_table_recovery,
        )
    try:
        import fitz  # type: ignore

        document = fitz.open(file_path)
        page_tables = _extract_tables_from_pdf(file_path) if enable_table_recovery else {}
        pages: list[PageContent] = []
        for page_index, page in enumerate(document, start=1):
            page_text = _normalize_text(page.get_text("text"))
            if enable_ocr and _needs_ocr(page_text, page):
                ocr_text = _normalize_text(_extract_ocr_text(page))
                if len(ocr_text) > len(page_text):
                    page_text = ocr_text
            table_texts = page_tables.get(page_index, [])
            if table_texts:
                table_blocks = [f"[表格]\n{table}" for table in table_texts]
                parts = [page_text] if page_text else []
                parts.extend(table_blocks)
                page_text = "\n\n".join(parts)
            pages.append(
                PageContent(
                    page_no=page_index,
                    text=page_text,
                    section_path=_detect_section_path(page_text, page_index),
                    chunk_type="prose",
                )
            )
        return _postprocess_pages(pages)
    except Exception:
        return _read_pdf_pages_with_pdfplumber(
            file_path,
            enable_table_recovery=enable_table_recovery,
        )


def _is_chunk_boundary_line(line: str) -> bool:
    """判断一行是否是 chunk 切分的硬边界（主条款/章节标题）。

    只将"第X条/章/节"作为硬边界——这些是独立的语义单元。
    分项标记（（一）、一、）改为软边界，不再独立成块，
    而是合并到所属主条款中，避免碎片化。
    """
    normalized = _normalize_line(line)
    if not normalized:
        return False
    boundary_patterns = [
        r"^第[\d一二三四五六七八九十百千万]+条",
        r"^第[\d一二三四五六七八九十百千万]+章",
        r"^第[\d一二三四五六七八九十百千万]+节",
    ]
    return any(re.match(pattern, normalized) for pattern in boundary_patterns)


def _is_sub_item_line(line: str) -> bool:
    """判断一行是否是条款下的分项标记（（一）、一、2. 等）。

    分项不作为硬切分边界，而是合并到所属主条款 chunk。
    """
    normalized = _normalize_line(line)
    if not normalized:
        return False
    sub_item_patterns = [
        r"^[一二三四五六七八九十]+、",
        r"^（[一二三四五六七八九十]+）",
        r"^\([一二三四五六七八九十]+\)",
        r"^第?[（(]?\d+[)）.、]",
    ]
    return any(re.match(pattern, normalized) for pattern in sub_item_patterns)


def _segment_page_parts(text: str) -> list[str]:
    lines = [_normalize_line(line) for line in text.splitlines() if _normalize_line(line)]
    if not lines:
        return []

    grouped_parts: list[str] = []
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("[表格]"):
            if current_lines:
                grouped_parts.append("\n".join(current_lines))
                current_lines = []
            grouped_parts.append(line)
            continue
        if current_lines and _is_chunk_boundary_line(line):
            grouped_parts.append("\n".join(current_lines))
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_lines:
        grouped_parts.append("\n".join(current_lines))

    parts: list[str] = []
    for part in grouped_parts:
        normalized_part = _normalize_text(part)
        if not normalized_part:
            continue
        if normalized_part.startswith("[表格]") or _is_chunk_boundary_line(normalized_part):
            parts.append(normalized_part)
            continue
        parts.extend(
            _normalize_text(segment)
            for segment in re.split(r"(?<=。)|(?<=；)|(?<=\n{2})", normalized_part)
            if _normalize_text(segment)
        )
    return parts


# 单个 chunk 文本的目标上限。超过此值的段落会按句子/自然段边界二次切分，
# 避免无脑截断（[:1200]）造成的内容丢失。
MAX_CHUNK_LENGTH = 1200
# 二次切分后每个子块的目标长度上限（留出拼接余量）。
SPLIT_TARGET_LENGTH = 1000


def _split_table_part(part: str) -> list[str]:
    """将超长表格块按行切分为多块，每块重复表头，避免 [:1200] 硬截断丢数据。

    年报表格（如资产负债表、利润表）与募集说明书表格常超过 MAX_CHUNK_LENGTH，
    原逻辑直接截断会丢失表格后半部分（含总资产、净利润等关键行），对数值类题目
    是致命的。改为按行累积切分，每块不超过 SPLIT_TARGET_LENGTH，且每块开头重复
    [表格] 标记与表头行，保证任意一块都能独立命中检索。
    """
    if "\n" not in part:
        return [part[:MAX_CHUNK_LENGTH]]
    header_end = part.index("\n")
    header_line = part[:header_end]  # 通常是 "[表格]"
    body_lines = part[header_end + 1 :].split("\n")
    blocks: list[str] = []
    current_rows: list[str] = []
    current_len = 0
    for row in body_lines:
        row_len = len(row) + 1
        if current_rows and current_len + row_len > SPLIT_TARGET_LENGTH:
            blocks.append(f"{header_line}\n" + "\n".join(current_rows))
            current_rows = []
            current_len = 0
        current_rows.append(row)
        current_len += row_len
    if current_rows:
        blocks.append(f"{header_line}\n" + "\n".join(current_rows))
    return blocks or [part[:MAX_CHUNK_LENGTH]]


def _split_oversized_part(part: str) -> list[str]:
    """将超过 MAX_CHUNK_LENGTH 的段落按句子边界拆分为多块。

    优先在句号/分号/换行处切分，使每个子块不超过 SPLIT_TARGET_LENGTH。
    无法按自然边界切分时退化为固定长度截断（仍优于直接丢弃）。
    表格块按行切分（见 _split_table_part），条款块（以第X条开头）按句切分。
    """
    if len(part) <= MAX_CHUNK_LENGTH:
        return [part]
    if part.startswith("[表格]"):
        return _split_table_part(part)
    # 优先按双换行（自然段）拆分；Python 3.7+ 的 re 不支持变长 lookbehind，
    # 改用 split 保留分隔符的方式实现"在 \n\n 处切分但保留换行"。
    segments = re.split(r"(\n{2,})", part)
    if len(segments) > 1:
        # 将分隔符并回前一段
        merged_segs: list[str] = []
        for i in range(0, len(segments) - 1, 2):
            merged_segs.append(segments[i] + (segments[i + 1] if i + 1 < len(segments) else ""))
        if segments[-1].strip():
            merged_segs.append(segments[-1])
        segments = merged_segs
    if len(segments) == 1:
        # 没有自然段边界，按句号/分号拆分
        segments = re.split(r"(?<=[。；])", part)
    blocks: list[str] = []
    current = ""
    for seg in segments:
        if not seg.strip():
            continue
        if len(current) + len(seg) <= SPLIT_TARGET_LENGTH:
            current += seg
        else:
            if current:
                blocks.append(current)
            # 单个 segment 自身超长时硬截断兜底
            current = seg if len(seg) <= MAX_CHUNK_LENGTH else seg[:MAX_CHUNK_LENGTH]
    if current:
        blocks.append(current)
    return blocks or [part[:MAX_CHUNK_LENGTH]]


def _split_page_text(page: PageContent, document_id: str, title: str, domain: str, chunk_index_start: int) -> list[StructuredChunk]:
    paragraphs = _segment_page_parts(page.text)
    # 聚合碎片：将短段落合并到前一个块，目标长度不超过 SPLIT_TARGET_LENGTH。
    # 原阈值（前段<60 且 当段<120）太低，导致 88% 切片短于 200 字。
    # 新策略：只要当前块未达到目标长度就继续追加，且不跨越条款/表格边界。
    merged_paragraphs: list[str] = []
    for part in paragraphs:
        is_boundary = part.startswith("[表格]") or _is_chunk_boundary_line(part)
        if (
            merged_paragraphs
            and not is_boundary
            and len(merged_paragraphs[-1]) < SPLIT_TARGET_LENGTH
        ):
            merged_paragraphs[-1] = _normalize_text(f"{merged_paragraphs[-1]} {part}")
        else:
            merged_paragraphs.append(part)

    # 对超过上限的块按句子边界二次切分，避免硬截断丢失内容
    final_parts: list[str] = []
    for part in merged_paragraphs:
        final_parts.extend(_split_oversized_part(part))

    chunks: list[StructuredChunk] = []
    seen_parts: set[str] = set()
    chunk_seq = 0
    for part in final_parts:
        dedupe_key = re.sub(r"\s+", "", part[:160])
        if dedupe_key in seen_parts:
            continue
        seen_parts.add(dedupe_key)
        clause_no = _extract_clause_no(part)
        chunk_type = "table" if part.startswith("[表格]") else ("clause" if clause_no else page.chunk_type)
        chunks.append(
            StructuredChunk(
                chunkId=f"{document_id}-chunk-{chunk_index_start + chunk_seq}",
                docId=document_id,
                title=title,
                domain=domain,
                pageNo=page.page_no,
                sectionPath=page.section_path,
                chunkType=chunk_type,
                clauseNo=clause_no,
                chunkText=part,
            )
        )
        chunk_seq += 1
    return chunks


def _fallback_chunk(document_id: str, title: str, domain: str, message: str) -> list[StructuredChunk]:
    return [
        StructuredChunk(
            chunkId=f"{document_id}-chunk-1",
            docId=document_id,
            title=title,
            domain=domain,
            pageNo=1,
            sectionPath="默认章节",
            chunkType="prose",
            clauseNo=None,
            chunkText=message,
        )
    ]


def _read_txt_pages(file_path: Path) -> list[PageContent]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    normalized = _normalize_text(text)
    return [PageContent(page_no=1, text=normalized, section_path=_detect_section_path(normalized, 1))]


def preprocess_document(
    record: DocumentRecord,
    domain: str,
    *,
    enable_ocr: bool = True,
    enable_table_recovery: bool = True,
    prefer_pdfplumber: bool = False,
) -> list[StructuredChunk]:
    file_path = Path(record.source_path)
    if not file_path.exists():
        return _fallback_chunk(record.document_id, record.title, domain, f"未找到源文件：{record.source_path}")

    if record.file_type == "txt":
        pages = _read_txt_pages(file_path)
    elif record.file_type == "html":
        pages = _read_html_pages(file_path)
    else:
        pages = _read_pdf_pages(
            file_path,
            enable_ocr=enable_ocr,
            enable_table_recovery=enable_table_recovery,
            prefer_pdfplumber=prefer_pdfplumber,
        )

    if not pages:
        return _fallback_chunk(record.document_id, record.title, domain, "未能解析出正文内容，请检查源文件。")

    chunks: list[StructuredChunk] = []
    chunk_index = 1
    for page in pages:
        if not page.text.strip():
            continue
        page_chunks = _split_page_text(page, record.document_id, record.title, domain, chunk_index)
        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    if not chunks:
        return _fallback_chunk(record.document_id, record.title, domain, "未能解析出正文内容，请检查源文件。")
    return chunks


def create_document_record(file_name: str, file_type: str, source_path: str | None) -> DocumentRecord:
    return DocumentRecord(
        documentId=str(uuid.uuid4()),
        title=file_name,
        fileType=file_type,
        sourcePath=source_path or "",
        status="queued",
        chunkCount=0,
    )
