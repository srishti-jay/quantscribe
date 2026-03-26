"""
Text cleaning utilities for Indian financial PDFs.

Handles:
- Zero-width unicode garbage
- Indian currency formatting (₹ 1,23,456.78)
- Accounting negatives in parentheses
- Whitespace normalization
"""

from __future__ import annotations

import re

# ── Unicode garbage commonly found in Indian bank PDFs ──
UNICODE_GARBAGE = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u00ad",  # Soft hyphen
    "\ufeff",  # BOM / zero-width no-break space
    "\u00a0",  # Non-breaking space (replace with regular space)
    "\u2028",  # Line separator
    "\u2029",  # Paragraph separator
    "\uf0b7",  # Private use area (common in PDF bullet points)
    "\uf0a7",  # Private use area (common in PDF symbols)
]

# ── Indian currency regex ──
INDIAN_NUMBER_RE = re.compile(
    r"₹?\s*\(?\d{1,3}(?:,\d{2})*(?:,\d{3})?(?:\.\d+)?\)?"
)


def strip_unicode_garbage(text: str) -> str:
    """
    Remove zero-width and invisible unicode characters.
    Normalize whitespace to single spaces.

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned text with normalized whitespace.
    """
    for char in UNICODE_GARBAGE:
        text = text.replace(char, " " if char == "\u00a0" else "")

    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Preserve paragraph breaks (double newline) but normalize singles
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def normalize_indian_currency(text: str) -> str:
    """
    Convert Indian-format numbers to parseable floats.

    Examples:
        '₹ 1,23,456.78' -> '123456.78'
        '(1,234.56)' -> '-1234.56'  (parentheses = negative in accounting)
        '₹ 45,00,000' -> '4500000'
        '12.5%' -> '12.5' (percentage sign stripped)

    Args:
        text: A string containing an Indian-formatted number.

    Returns:
        String representation of the parsed float, or the original text if unparseable.
    """
    text = text.replace("₹", "").replace(" ", "").strip()

    # Strip percentage for parsing (caller handles unit separately)
    text = text.rstrip("%")

    # Handle accounting negatives: (1,234.56) -> -1234.56
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1]

    # Remove Indian comma formatting
    text = text.replace(",", "")

    # Handle common textual representations
    text_lower = text.lower().strip()
    if text_lower in ("nil", "n/a", "-", "--", "—", ""):
        return "0"

    try:
        value = float(text)
        if is_negative:
            value = -value
        return str(value)
    except ValueError:
        return text  # Return original if not parseable


def clean_table_cell(cell: str | None) -> str:
    """
    Clean a single table cell value.

    Handles None (from merged cells), strips whitespace,
    removes unicode garbage, and normalizes numbers.

    Args:
        cell: Raw cell value from pdfplumber/camelot.

    Returns:
        Cleaned cell string. Empty string for None/empty cells.
    """
    if cell is None:
        return ""

    cell = str(cell).strip()
    cell = strip_unicode_garbage(cell)

    # Try to normalize if it looks like a number
    if INDIAN_NUMBER_RE.fullmatch(cell.strip()):
        cell = normalize_indian_currency(cell)

    return cell


def forward_fill_none(table: list[list[str | None]]) -> list[list[str]]:
    """
    Forward-fill None values in a table (handles merged cells).

    In pdfplumber output, merged cells appear as None in subsequent rows.
    This fills them with the value from the cell above.

    Args:
        table: 2D list of cell values (rows x columns).

    Returns:
        Table with None values filled from the cell above.
    """
    if not table:
        return []

    result: list[list[str]] = []
    num_cols = max(len(row) for row in table)

    for row_idx, row in enumerate(table):
        # Pad short rows
        padded = list(row) + [None] * (num_cols - len(row))
        cleaned_row: list[str] = []

        for col_idx, cell in enumerate(padded):
            if cell is None and row_idx > 0 and col_idx < len(result[-1]):
                # Forward fill from the row above
                cleaned_row.append(result[-1][col_idx])
            else:
                cleaned_row.append(clean_table_cell(cell))

        result.append(cleaned_row)

    return result
