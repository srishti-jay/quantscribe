"""
Structured logging configuration for QuantScribe.

Every ETL failure, LLM retry, and evaluation result is logged
as structured JSON for post-hoc analysis.

Usage:
    from quantscribe.logging_config import get_logger
    logger = get_logger("quantscribe.etl")
    logger.warn("pdfplumber_table_miss", page=42, bank="HDFC_BANK")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

_CONFIGURED = False


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Configure structured logging for the entire application. Call once at startup."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # ── Configure structlog ──
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=open(log_path / "quantscribe.jsonl", "a")  # noqa: SIM115
        ),
        cache_logger_on_first_use=True,
    )

    # ── Also log to stderr for development ──
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.root.addHandler(console_handler)
    logging.root.setLevel(level)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given module name."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
