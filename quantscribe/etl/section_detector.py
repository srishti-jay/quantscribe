"""
Section header detection for PDF pages.

Uses three strategies in priority order:
1. Fuzzy match against KNOWN_SECTIONS (from config)
2. Font-size heuristic: text blocks with font size > 1.3x page median
3. Position heuristic: bold text in the top 15% of the page

TODO: Implement after page classifier is calibrated on sample PDFs.
"""

from __future__ import annotations

from typing import Optional

from difflib import get_close_matches

from quantscribe.config import KNOWN_SECTIONS
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.etl.section_detector")


def detect_section_header(
    page_blocks: list[dict],
    page_number: int,
) -> Optional[str]:
    """
    Detect the section header for a page.

    Args:
        page_blocks: List of text block dicts from PyMuPDF, each with:
            - text: str
            - font_size: Optional[float]
            - median_font_size: Optional[float]
            - y_position: Optional[float]
            - page_height: Optional[float]
            - is_bold: Optional[bool]
        page_number: Page number (for logging).

    Returns:
        Best matching section header string, or None.
    """
    candidates: list[str] = []

    for block in page_blocks:
        text = block.get("text", "").strip()
        if not text or len(text) > 100:
            continue

        # Strategy 1: Known section matching (highest priority)
        matches = get_close_matches(text, KNOWN_SECTIONS, n=1, cutoff=0.75)
        if matches:
            logger.info("section_header_detected", page=page_number, header=matches[0], strategy="known_match")
            return matches[0]

        # Strategy 2: Font-size heuristic
        font_size = block.get("font_size")
        median_size = block.get("median_font_size")
        if font_size and median_size and font_size > median_size * 1.3:
            candidates.append(text)

        # Strategy 3: Position heuristic (top 15% of page, bold)
        y_pos = block.get("y_position")
        page_h = block.get("page_height")
        if y_pos and page_h and y_pos < page_h * 0.15:
            if block.get("is_bold"):
                candidates.append(text)

    if candidates:
        logger.info("section_header_detected", page=page_number, header=candidates[0], strategy="heuristic")
        return candidates[0]

    logger.info("section_header_not_found", page=page_number)
    return None
