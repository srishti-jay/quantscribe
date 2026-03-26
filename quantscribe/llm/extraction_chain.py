"""
LLM extraction chain with Pydantic output parsing and retry logic.

Uses LangChain + Gemini API to extract structured ThematicExtraction
objects from retrieved financial text.

TODO: Implement in Phase 4 after retrieval pipeline is validated.
"""

from __future__ import annotations


def build_extraction_chain(llm, max_retries: int = 3):  # type: ignore[no-untyped-def]
    """
    Build a LangChain extraction chain with:
    1. Pydantic output parsing (ThematicExtraction schema)
    2. Automatic retry on malformed JSON
    3. Citation validation (excerpt must appear in context)
    4. Structured error logging

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("Phase 4: Awaiting validated retrieval pipeline.")
