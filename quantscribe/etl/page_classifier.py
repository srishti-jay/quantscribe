"""
Page classifier for PDF pages.

Classifies each page as NARRATIVE, TABULAR, MIXED, or GRAPHICAL
using pdfplumber table detection + PyMuPDF text block analysis.

Thresholds calibrated on HDFC Bank FY25 and SBI FY25 annual reports.
"""

from __future__ import annotations

from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from quantscribe.schemas.etl import PageType, ParsedPage
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.etl.page_classifier")

# ── Thresholds (calibrated on HDFC FY25 + SBI FY25) ──
MIN_WORDS_FOR_CONTENT = 50         # Below this → GRAPHICAL (chart-only pages)
MIN_NARRATIVE_WORDS_FOR_MIXED = 80 # Words OUTSIDE tables needed to classify as MIXED


def classify_page(page_number: int, pdf_path: str) -> ParsedPage:
    """
    Classify a single PDF page and return a ParsedPage.

    Two-pass approach:
      Pass 1 (pdfplumber): detect tables and their bounding boxes.
      Pass 2 (PyMuPDF):    detect text blocks and images.

    Then compute how much text lives OUTSIDE table regions
    to distinguish TABULAR from MIXED pages.

    Args:
        page_number: 0-indexed page number.
        pdf_path:    Absolute path to the PDF file.

    Returns:
        ParsedPage with page_type, raw_text (for narrative),
        tables (for tabular), and confidence_score.
    """
    # ── Pass 1: pdfplumber — table detection ──
    with pdfplumber.open(pdf_path) as pdf:
        if page_number >= len(pdf.pages):
            return _empty_page(page_number)
        plumber_page = pdf.pages[page_number]
        detected_tables = plumber_page.find_tables()
        table_count = len(detected_tables)

        # Extract table bounding boxes for later filtering
        table_bboxes = []
        extracted_tables: list[list[list[str | None]]] = []
        for tbl in detected_tables:
            table_bboxes.append(tbl.bbox)  # (x0, y0, x1, y1)
            extracted_tables.append(tbl.extract())

    # ── Pass 2: PyMuPDF — text blocks and images ──
    doc = fitz.open(pdf_path)
    mu_page = doc[page_number]
    text_blocks = mu_page.get_text("blocks")
    all_text = mu_page.get_text()
    total_word_count = len(all_text.split())
    image_count = len(mu_page.get_images())
    doc.close()

    # ── Compute narrative text (words OUTSIDE table bounding boxes) ──
    narrative_words = _count_words_outside_tables(text_blocks, table_bboxes)

    # ── Decision tree ──
    page_type, confidence = _decide_page_type(
        table_count=table_count,
        total_words=total_word_count,
        narrative_words=narrative_words,
        image_count=image_count,
    )

    # ── Build ParsedPage ──
    raw_text: Optional[str] = None
    tables_as_dicts: Optional[list[dict]] = None
    warnings: list[str] = []

    if page_type in (PageType.NARRATIVE, PageType.MIXED):
        raw_text = all_text.strip() if all_text else None
        if not raw_text:
            warnings.append("empty_text_on_narrative_page")

    if page_type in (PageType.TABULAR, PageType.MIXED):
        tables_as_dicts = _tables_to_dicts(extracted_tables)
        if not tables_as_dicts:
            warnings.append("no_tables_extracted")

    if page_type == PageType.GRAPHICAL:
        warnings.append("graphical_page_skipped")

    logger.info(
        "page_classified",
        page=page_number + 1,
        page_type=page_type.value,
        tables=table_count,
        total_words=total_word_count,
        narrative_words=narrative_words,
        images=image_count,
        confidence=round(confidence, 2),
    )

    return ParsedPage(
        page_number=page_number + 1,  # Convert to 1-indexed for metadata
        page_type=page_type,
        raw_text=raw_text,
        tables=tables_as_dicts,
        extraction_warnings=warnings,
        confidence_score=confidence,
    )


def classify_all_pages(pdf_path: str) -> list[ParsedPage]:
    """Classify every page in a PDF. Returns list of ParsedPage objects."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    results = []
    for pg in range(total_pages):
        try:
            result = classify_page(pg, pdf_path)
            results.append(result)
        except Exception as e:
            logger.error("page_classification_failed", page=pg + 1, error=str(e))
            results.append(_empty_page(pg))

    # Summary
    counts = {}
    for r in results:
        counts[r.page_type.value] = counts.get(r.page_type.value, 0) + 1
    logger.info("classification_complete", total_pages=total_pages, counts=counts)

    return results


# ── Internal helpers ──


def _decide_page_type(
    table_count: int,
    total_words: int,
    narrative_words: int,
    image_count: int,
) -> tuple[PageType, float]:
    """
    Decision tree for page classification.

    Returns (PageType, confidence_score).
    """
    # Very few words → likely a graphical/decorative page
    if total_words < MIN_WORDS_FOR_CONTENT:
        return PageType.GRAPHICAL, 0.85

    # No tables detected
    if table_count == 0:
        return PageType.NARRATIVE, 0.90

    # Tables detected — check if there's also substantial narrative
    if narrative_words >= MIN_NARRATIVE_WORDS_FOR_MIXED:
        return PageType.MIXED, 0.80

    # Tables with little outside text → pure tabular
    return PageType.TABULAR, 0.85


def _count_words_outside_tables(
    text_blocks: list,
    table_bboxes: list[tuple],
) -> int:
    """
    Count words in text blocks whose centroids fall OUTSIDE any table bbox.

    This separates narrative content from table content on mixed pages.
    """
    if not table_bboxes:
        # No tables — all text is narrative
        return sum(
            len(b[4].split()) for b in text_blocks if b[6] == 0  # type 0 = text
        )

    outside_words = 0
    for block in text_blocks:
        if block[6] != 0:  # Skip image blocks
            continue
        # Block bbox: (x0, y0, x1, y1, text, block_no, block_type)
        bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
        centroid_x = (bx0 + bx1) / 2
        centroid_y = (by0 + by1) / 2

        inside_table = False
        for tx0, ty0, tx1, ty1 in table_bboxes:
            if tx0 <= centroid_x <= tx1 and ty0 <= centroid_y <= ty1:
                inside_table = True
                break

        if not inside_table:
            outside_words += len(block[4].split())

    return outside_words


def _tables_to_dicts(raw_tables: list[list[list[str | None]]]) -> list[dict]:
    """
    Convert pdfplumber raw table output to list-of-dicts format.

    Each table becomes a list of row-dicts where keys are column headers.
    Returns a flat list of all row-dicts across all tables.
    """
    all_rows: list[dict] = []

    for table in raw_tables:
        if not table or len(table) < 2:
            continue

        # First row = headers
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]

        for row in table[1:]:
            row_dict = {}
            for i, cell in enumerate(row):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row_dict[key] = str(cell).strip() if cell else ""
            all_rows.append(row_dict)

    return all_rows


def _empty_page(page_number: int) -> ParsedPage:
    """Return a GRAPHICAL ParsedPage for failed/empty pages."""
    return ParsedPage(
        page_number=page_number + 1,
        page_type=PageType.GRAPHICAL,
        raw_text=None,
        tables=None,
        extraction_warnings=["page_parse_failed_or_empty"],
        confidence_score=0.0,
    )