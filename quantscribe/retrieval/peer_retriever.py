"""
PeerGroupRetriever: Fan-out retrieval across multiple BankIndex instances.

This is the primary mechanism preventing cross-entity contamination.
Each bank's index is queried independently, guaranteeing equal representation.
"""

from __future__ import annotations

import numpy as np

from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.retrieval.peer_retriever")


class PeerGroupRetriever:
    """
    Fan-out retrieval across multiple BankIndex instances.

    Guarantees each bank gets equal representation in results,
    preventing one bank's verbose disclosures from dominating.
    """

    def __init__(self, bank_indices: dict[str, BankIndex]):
        """
        Args:
            bank_indices: Mapping of index_name -> BankIndex.
        """
        self.bank_indices = bank_indices
        logger.info("peer_retriever_init", num_indices=len(bank_indices))

    def retrieve(
        self,
        query_vector: np.ndarray,
        peer_group: list[str],
        top_k_per_bank: int = 5,
    ) -> dict[str, list[dict]]:
        """
        Fan out the query to each bank's index independently.

        Args:
            query_vector: L2-normalized query vector, shape (1, dimension).
            peer_group: List of bank names to query.
            top_k_per_bank: Number of results per bank.

        Returns:
            Dict mapping bank_name -> list of result dicts,
            each with 'metadata' and 'score'.
        """
        results: dict[str, list[dict]] = {}

        for index_name, bank_index in self.bank_indices.items():
            if bank_index.size == 0:
                continue

            # Extract bank_name from the first metadata entry
            bank_name = bank_index.metadata_store[0].get("bank_name", "")

            if bank_name in peer_group:
                bank_results = bank_index.search(query_vector, top_k=top_k_per_bank)
                results[bank_name] = bank_results
                logger.info(
                    "bank_retrieved",
                    bank=bank_name,
                    results=len(bank_results),
                    top_score=bank_results[0]["score"] if bank_results else 0.0,
                )

        # Warn about banks with no results
        for bank in peer_group:
            if bank not in results:
                logger.warn("bank_not_found_in_indices", bank=bank)

        return results

    def list_available_banks(self) -> list[str]:
        """Return list of bank names available in loaded indices."""
        banks = set()
        for bank_index in self.bank_indices.values():
            if bank_index.metadata_store:
                banks.add(bank_index.metadata_store[0].get("bank_name", ""))
        return sorted(banks)
