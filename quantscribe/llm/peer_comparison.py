"""
Peer comparison orchestrator.

End-to-end pipeline:
1. Embed the thematic query
2. Fan-out retrieval to each bank's index
3. Format retrieved chunks with rigid delimiters
4. Run LLM extraction per bank
5. Rank and synthesize

TODO: Implement in Phase 4 after extraction chain is validated.
"""

from __future__ import annotations


def run_peer_comparison(
    theme: str,
    peer_group: list[str],
    retriever,  # type: ignore[no-untyped-def]
    embedding_pipeline,  # type: ignore[no-untyped-def]
    extraction_chain,  # type: ignore[no-untyped-def]
    top_k_per_bank: int = 5,
):  # type: ignore[no-untyped-def]
    """
    Run a full peer comparison for a given theme.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("Phase 4: Awaiting validated extraction chain.")
