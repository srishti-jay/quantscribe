"""
DeepEval evaluation module.

Measures LLM output faithfulness and answer relevancy
using DeepEval framework with Gemini as the judge LLM.

DeepEval defaults to OpenAI. This module provides a proper
DeepEvalBaseLLM subclass that wraps Gemini, so no OpenAI key is needed.

Usage:
    from quantscribe.evaluation.deepeval_eval import run_deepeval_evaluation

    results = run_deepeval_evaluation(
        theme="credit_risk",
        bank_name="HDFC_BANK",
        query="credit risk gross NPA net NPA provision coverage",
        retrieved_contexts=["chunk1 text...", "chunk2 text..."],
        llm_response="HDFC Bank's Gross NPA ratio stood at 1.33%...",
    )
"""

from __future__ import annotations

from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.evaluation.deepeval")


def _build_gemini_judge():
    """
    Build a Gemini-backed judge model for DeepEval.

    DeepEval requires a DeepEvalBaseLLM subclass with
    generate(), a_generate(), get_model_name(), and load_model().
    """
    from deepeval.models.base_model import DeepEvalBaseLLM
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()

    class GeminiJudge(DeepEvalBaseLLM):
        def __init__(self):
            self.model_name = settings.llm_model
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=0.0,
                google_api_key=settings.google_api_key,
            )

        def load_model(self):
            return self.llm

        def generate(self, prompt: str, **kwargs) -> str:
            response = self.llm.invoke(prompt)
            return response.content

        async def a_generate(self, prompt: str, **kwargs) -> str:
            response = await self.llm.ainvoke(prompt)
            return response.content

        def get_model_name(self) -> str:
            return self.model_name

    return GeminiJudge()


def run_deepeval_evaluation(
    theme: str,
    bank_name: str,
    query: str,
    retrieved_contexts: list[str],
    llm_response: str,
) -> dict[str, float]:
    """
    Run DeepEval faithfulness and answer relevancy evaluation.

    Uses Gemini as the judge LLM via GeminiJudge wrapper.

    Args:
        theme: The macro theme queried.
        bank_name: Bank being evaluated.
        query: The retrieval query text.
        retrieved_contexts: List of chunk texts sent to the LLM.
        llm_response: The LLM's output text.

    Returns:
        Dict with "faithfulness" and "answer_relevancy" scores (0.0 to 1.0).
        Returns -1.0 for any metric that fails to compute.
    """
    try:
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        logger.error("deepeval_import_failed", error=str(e))
        return {"faithfulness": -1.0, "answer_relevancy": -1.0}

    # ── Build Gemini judge ──
    try:
        judge = _build_gemini_judge()
    except Exception as e:
        logger.error("gemini_judge_init_failed", error=str(e)[:200])
        return {"faithfulness": -1.0, "answer_relevancy": -1.0}

    # ── Build test case ──
    test_case = LLMTestCase(
        input=query,
        actual_output=llm_response,
        retrieval_context=retrieved_contexts,
    )

    results: dict[str, float] = {}

    # ── Faithfulness ──
    try:
        faithfulness = FaithfulnessMetric(
            threshold=0.85,
            model=judge,
        )
        faithfulness.measure(test_case)
        results["faithfulness"] = round(float(faithfulness.score), 4)
        logger.info(
            "deepeval_faithfulness",
            bank=bank_name,
            theme=theme,
            score=results["faithfulness"],
            reason=faithfulness.reason[:200] if faithfulness.reason else "N/A",
        )
    except Exception as e:
        logger.error(
            "deepeval_faithfulness_failed",
            error=str(e)[:300],
            bank=bank_name,
        )
        results["faithfulness"] = -1.0

    # ── Answer Relevancy ──
    try:
        relevancy = AnswerRelevancyMetric(
            threshold=0.80,
            model=judge,
        )
        relevancy.measure(test_case)
        results["answer_relevancy"] = round(float(relevancy.score), 4)
        logger.info(
            "deepeval_answer_relevancy",
            bank=bank_name,
            theme=theme,
            score=results["answer_relevancy"],
            reason=relevancy.reason[:200] if relevancy.reason else "N/A",
        )
    except Exception as e:
        logger.error(
            "deepeval_answer_relevancy_failed",
            error=str(e)[:300],
            bank=bank_name,
        )
        results["answer_relevancy"] = -1.0

    return results


def run_deepeval_batch(
    evaluations: list[dict],
) -> list[dict]:
    """
    Run DeepEval evaluation on a batch of extractions.

    Args:
        evaluations: List of dicts, each with keys:
            - theme, bank_name, query, retrieved_contexts, llm_response

    Returns:
        List of result dicts with DeepEval scores appended.
    """
    all_results = []

    for i, eval_input in enumerate(evaluations):
        logger.info(
            "deepeval_batch_progress",
            index=i + 1,
            total=len(evaluations),
            bank=eval_input["bank_name"],
        )

        scores = run_deepeval_evaluation(
            theme=eval_input["theme"],
            bank_name=eval_input["bank_name"],
            query=eval_input["query"],
            retrieved_contexts=eval_input["retrieved_contexts"],
            llm_response=eval_input["llm_response"],
        )

        result = {**eval_input, **scores}
        all_results.append(result)

    return all_results
    