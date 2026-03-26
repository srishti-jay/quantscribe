"""
Numerical accuracy evaluator.

Compares extracted metrics against gold-standard test cases
with configurable relative tolerance.
"""

from __future__ import annotations

from quantscribe.schemas.extraction import ThematicExtraction
from quantscribe.schemas.evaluation import EvalTestCase
from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.evaluation.numerical")


def evaluate_numerical_accuracy(
    extracted: ThematicExtraction,
    gold: EvalTestCase,
    tolerance: float | None = None,
) -> dict[str, bool]:
    """
    Compare each extracted metric against gold standard.

    Args:
        extracted: LLM extraction output.
        gold: Ground-truth test case.
        tolerance: Relative tolerance (default: 0.5% from config).

    Returns:
        Dict mapping metric_name -> bool (True if within tolerance).
    """
    settings = get_settings()
    tol = tolerance or settings.numerical_tolerance

    # Build lookup from extracted metrics
    extracted_map: dict[str, float | None] = {
        m.metric_name: m.metric_value
        for m in extracted.extracted_metrics
    }

    results: dict[str, bool] = {}

    for metric_name, expected_value in gold.expected_metrics.items():
        if metric_name not in extracted_map:
            logger.warn(
                "metric_not_extracted",
                metric=metric_name,
                bank=gold.bank_name,
                test_id=gold.test_id,
            )
            results[metric_name] = False
            continue

        actual = extracted_map[metric_name]
        if actual is None:
            results[metric_name] = False
            continue

        if expected_value == 0:
            match = actual == 0
        else:
            relative_error = abs(actual - expected_value) / abs(expected_value)
            match = relative_error <= tol

        results[metric_name] = match

        if not match:
            logger.warn(
                "numerical_mismatch",
                metric=metric_name,
                expected=expected_value,
                actual=actual,
                test_id=gold.test_id,
            )

    accuracy = sum(results.values()) / max(len(results), 1)
    logger.info(
        "numerical_accuracy",
        test_id=gold.test_id,
        accuracy=accuracy,
        total_metrics=len(results),
        passed=sum(results.values()),
    )

    return results
