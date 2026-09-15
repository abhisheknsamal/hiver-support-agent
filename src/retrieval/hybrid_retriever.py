"""
Hybrid Historical Evidence Retriever
------------------------------------

Orchestrates the complete retrieval pipeline:

    Customer Query
          |
     ┌────┴────┐
     ↓         ↓
   BM25      BGE-large
     ↓         ↓
     └────┬────┘
          ↓
      RRF Fusion
          ↓
    RRF Candidate Pool
          ↓
    Cross-Encoder
          ↓
    Final Evidence

Runtime design:
    - BM25 and BGE independently retrieve candidates.
    - RRF combines their rankings.
    - Cross-Encoder performs final relevance reranking.
    - Evaluation-only fields are never returned.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from corpus import load_corpus
from bm25_retriever import BM25Retriever
from vector_retriever import BGEVectorRetriever
from rrf import reciprocal_rank_fusion
from reranker import CrossEncoderReranker


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BM25_K = 50
DEFAULT_BGE_K = 50
DEFAULT_RRF_K = 60
DEFAULT_FINAL_K = 5


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:
    """
    Complete hybrid historical evidence retrieval system.

    Pipeline:

        BM25 + BGE
             ↓
            RRF
             ↓
        Cross-Encoder
             ↓
        Final Evidence
    """

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        embedding_path: str,
        bm25_k: int = DEFAULT_BM25_K,
        bge_k: int = DEFAULT_BGE_K,
        rrf_k: int = DEFAULT_RRF_K,
        final_k: int = DEFAULT_FINAL_K,
        vector_model=None,
        reranker_model=None,
    ):
        """
        Initialize the complete hybrid retriever.

        Args:
            corpus_df:
                Structured historical resolution corpus.

            embedding_path:
                Path to precomputed BGE-large corpus embeddings.

            bm25_k:
                Number of BM25 candidates.

            bge_k:
                Number of BGE candidates.

            rrf_k:
                RRF smoothing parameter.

            final_k:
                Number of final evidence records.

            vector_model:
                Optional injected BGE model.

            reranker_model:
                Optional injected cross-encoder model.
        """

        if corpus_df.empty:
            raise ValueError(
                "Historical corpus cannot be empty."
            )

        if bm25_k <= 0:
            raise ValueError(
                "bm25_k must be greater than zero."
            )

        if bge_k <= 0:
            raise ValueError(
                "bge_k must be greater than zero."
            )

        if rrf_k < 0:
            raise ValueError(
                "rrf_k must be non-negative."
            )

        if final_k <= 0:
            raise ValueError(
                "final_k must be greater than zero."
            )

        self.corpus_df = corpus_df.reset_index(
            drop=True
        ).copy()

        self.bm25_k = bm25_k
        self.bge_k = bge_k
        self.rrf_k = rrf_k
        self.final_k = final_k

        # ----------------------------------------------------
        # Retrieval components
        # ----------------------------------------------------

        self.bm25_retriever = BM25Retriever(
            corpus_df=self.corpus_df
        )

        self.bge_retriever = BGEVectorRetriever(
            corpus_df=self.corpus_df,
            embedding_path=embedding_path,
            model=vector_model,
        )

        self.reranker = CrossEncoderReranker(
            model=reranker_model
        )

    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
    ) -> Dict[str, object]:
        """
        Run the complete hybrid retrieval pipeline.

        Returns:

            query
            bm25_results
            bge_results
            rrf_results
            final_evidence
        """

        if not isinstance(query, str):
            query = str(query)

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        # ----------------------------------------------------
        # Stage 1: BM25
        # ----------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=self.bm25_k,
            )
        )

        # ----------------------------------------------------
        # Stage 2: BGE
        # ----------------------------------------------------

        bge_results = (
            self.bge_retriever.search(
                query=query,
                top_k=self.bge_k,
            )
        )

        # ----------------------------------------------------
        # Stage 3: RRF
        # ----------------------------------------------------

        rrf_results = (
            reciprocal_rank_fusion(
                bm25_results=bm25_results,
                bge_results=bge_results,
                k=self.rrf_k,
                top_k=self.bm25_k + self.bge_k,
            )
        )

        # ----------------------------------------------------
        # Stage 4: Cross-Encoder
        # ----------------------------------------------------

        final_evidence = (
            self.reranker.rerank(
                query=query,
                candidates=rrf_results,
                top_k=self.final_k,
            )
        )

        # ----------------------------------------------------
        # Runtime safety
        # ----------------------------------------------------

        forbidden_columns = {
            "golden_id",
            "golden_intent",
            "intent_match",
            "gold_automation",
            "gold_risk",
            "gold_resolution",
        }

        final_evidence = final_evidence[
            [
                column
                for column in final_evidence.columns
                if column not in forbidden_columns
            ]
        ]

        return {
            "query": query,
            "bm25_results": bm25_results,
            "bge_results": bge_results,
            "rrf_results": rrf_results,
            "final_evidence": final_evidence,
        }


# ============================================================
# SUMMARY
# ============================================================

def summarize_retrieval(
    result: Dict[str, object],
) -> Dict[str, object]:
    """
    Produce a compact retrieval summary suitable for logging.
    """

    bm25_results = result["bm25_results"]
    bge_results = result["bge_results"]
    rrf_results = result["rrf_results"]
    final_evidence = result["final_evidence"]

    return {
        "query": result["query"],
        "bm25_candidates": int(
            len(bm25_results)
        ),
        "bge_candidates": int(
            len(bge_results)
        ),
        "rrf_candidates": int(
            len(rrf_results)
        ),
        "final_evidence_count": int(
            len(final_evidence)
        ),
        "top_case_id": (
            final_evidence.iloc[0]["case_id"]
            if not final_evidence.empty
            else None
        ),
    }


# ============================================================
# LOCAL FAKE MODELS
# ============================================================

class FakeVectorModel:
    """
    Deterministic fake BGE model for integration testing.

    No external model download is required.
    """

    def encode(
        self,
        query,
        normalize_embeddings=True,
    ):
        import numpy as np

        text = str(query).lower()

        if "package" in text:
            vector = np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )

        elif "password" in text:
            vector = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            )

        else:
            vector = np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            )

        return vector


class FakeRerankerModel:
    """
    Deterministic fake cross-encoder for integration testing.
    """

    def predict(
        self,
        pairs,
    ):
        scores = []

        for query, document in pairs:

            query_words = set(
                str(query)
                .lower()
                .split()
            )

            document_words = set(
                str(document)
                .lower()
                .split()
            )

            overlap = len(
                query_words
                & document_words
            )

            scores.append(
                float(overlap)
            )

        return scores


# ============================================================
# LOCAL INTEGRATION TEST
# ============================================================

if __name__ == "__main__":

    import numpy as np

    print("=" * 100)
    print("HIVER SUPPORT AGENT — HYBRID RETRIEVER INTEGRATION TEST")
    print("=" * 100)

    # --------------------------------------------------------
    # Synthetic corpus
    # --------------------------------------------------------

    test_df = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-004",
            ],
            "customer_problem": [
                "Where is my package?",
                "My package delivery is delayed.",
                "I forgot my password.",
                "I need a refund.",
            ],
            "support_action": [
                "Provided package tracking guidance.",
                "Provided delivery delay guidance.",
                "Provided account recovery guidance.",
                "Explained refund options.",
            ],
            "resolution_type": [
                "ORDER_TRACKING",
                "ORDER_DELIVERY",
                "ACCOUNT_ACCESS",
                "RETURNS_REFUNDS",
            ],
            "evidence_quality": [
                "HIGH",
                "HIGH",
                "HIGH",
                "HIGH",
            ],
            "source": [
                "historical_support_case",
                "historical_support_case",
                "historical_support_case",
                "historical_support_case",
            ],
        }
    )

    test_df["retrieval_text"] = (
        "Customer problem: "
        + test_df["customer_problem"]
        + "\nSupport action: "
        + test_df["support_action"]
        + "\nResolution type: "
        + test_df["resolution_type"]
    )

    # --------------------------------------------------------
    # Synthetic BGE embeddings
    # --------------------------------------------------------

    test_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    temp_embedding_path = (
        Path(__file__).resolve().parent
        / "_test_hybrid_embeddings.npy"
    )

    np.save(
        temp_embedding_path,
        test_embeddings,
    )

    try:

        # ----------------------------------------------------
        # Initialize complete pipeline
        # ----------------------------------------------------

        retriever = HybridRetriever(
            corpus_df=test_df,
            embedding_path=str(
                temp_embedding_path
            ),
            bm25_k=3,
            bge_k=3,
            rrf_k=60,
            final_k=2,
            vector_model=FakeVectorModel(),
            reranker_model=FakeRerankerModel(),
        )

        print(
            "PASS | Hybrid retriever initialization"
        )

        # ----------------------------------------------------
        # Full retrieval
        # ----------------------------------------------------

        result = retriever.retrieve(
            "Where is my package?"
        )

        print(
            "PASS | BM25 stage"
        )

        print(
            "PASS | BGE stage"
        )

        print(
            "PASS | RRF stage"
        )

        print(
            "PASS | Cross-Encoder stage"
        )

        # ----------------------------------------------------
        # Validate result structure
        # ----------------------------------------------------

        assert (
            len(result["bm25_results"])
            <= 3
        )

        assert (
            len(result["bge_results"])
            <= 3
        )

        assert (
            len(result["rrf_results"])
            <= 6
        )

        assert (
            len(result["final_evidence"])
            <= 2
        )

        print(
            "PASS | End-to-end candidate flow"
        )

        # ----------------------------------------------------
        # Top evidence
        # ----------------------------------------------------

        final_evidence = result[
            "final_evidence"
        ]

        assert not final_evidence.empty

        assert (
            final_evidence.iloc[0]["case_id"]
            in {
                "CASE-001",
                "CASE-002",
            }
        )

        print(
            "PASS | Final historical evidence"
        )

        # ----------------------------------------------------
        # Runtime field isolation
        # ----------------------------------------------------

        forbidden_columns = {
            "golden_id",
            "golden_intent",
            "intent_match",
            "gold_automation",
            "gold_risk",
            "gold_resolution",
        }

        assert not (
            forbidden_columns
            & set(final_evidence.columns)
        )

        print(
            "PASS | Evaluation-only field isolation"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = summarize_retrieval(
            result
        )

        assert (
            summary["final_evidence_count"]
            <= 2
        )

        print(
            "PASS | Retrieval summary"
        )

        print()
        print("Retrieval summary:")
        print(summary)

        print()
        print("Top final evidence:")
        print(
            final_evidence.iloc[0].to_dict()
        )

        print("=" * 100)
        print("HYBRID RETRIEVER INTEGRATION TEST: PASS")
        print("=" * 100)

    finally:

        if temp_embedding_path.exists():
            temp_embedding_path.unlink()