"""
Cross-Encoder Evidence Reranker
--------------------------------

Final precision stage of the hybrid retrieval pipeline.

Pipeline:

    BM25
      \
       → RRF → Candidate Pool → Cross-Encoder → Final Evidence
      /
    BGE

Model:

    BAAI/bge-reranker-base

The cross-encoder scores the query and each candidate case together,
which provides more precise query-evidence relevance than independent
embedding similarity.

Runtime-safe design:
    - Evaluation-only fields are excluded.
    - Golden labels are never used during runtime reranking.
"""

from typing import List, Optional

import pandas as pd


# ============================================================
# OPTIONAL DEPENDENCY
# ============================================================

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RERANKER_MODEL = (
    "BAAI/bge-reranker-base"
)

DEFAULT_CANDIDATE_K = 50
DEFAULT_OUTPUT_K = 5


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "case_id",
    "customer_problem",
    "support_action",
    "resolution_type",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_candidates(
    candidates: pd.DataFrame,
) -> None:
    """
    Validate RRF candidate results before reranking.
    """

    missing = (
        REQUIRED_COLUMNS
        - set(candidates.columns)
    )

    if missing:
        raise ValueError(
            "Candidates are missing required columns: "
            + ", ".join(sorted(missing))
        )

    if candidates["case_id"].duplicated().any():
        raise ValueError(
            "Candidates contain duplicate case_id values."
        )


# ============================================================
# CANDIDATE TEXT
# ============================================================

def build_candidate_text(
    row: pd.Series,
) -> str:
    """
    Build the historical evidence text supplied to the
    cross-encoder.
    """

    return (
        "Customer problem: "
        + str(row["customer_problem"])
        + "\nSupport action: "
        + str(row["support_action"])
        + "\nResolution type: "
        + str(row["resolution_type"])
    )


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranker for historical support evidence.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model=None,
    ):
        """
        Initialize the reranker.

        Args:
            model_name:
                Hugging Face/SentenceTransformers cross-encoder model.

            model:
                Optional injected model for testing.
        """

        self.model_name = model_name

        if model is not None:
            self.model = model

        else:

            if CrossEncoder is None:
                raise ImportError(
                    "sentence-transformers is not installed.\n"
                    "Install it with:\n"
                    "pip install sentence-transformers"
                )

            self.model = CrossEncoder(
                model_name
            )

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query: str,
        candidates: pd.DataFrame,
        top_k: int = DEFAULT_OUTPUT_K,
    ) -> pd.DataFrame:
        """
        Rerank RRF candidates using the cross-encoder.

        Args:
            query:
                Current customer query.

            candidates:
                Candidate DataFrame produced by RRF.

            top_k:
                Number of final evidence items.

        Returns:
            Runtime-safe reranked DataFrame.
        """

        if not isinstance(query, str):
            query = str(query)

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        validate_candidates(
            candidates
        )

        if candidates.empty:
            return pd.DataFrame(
                columns=[
                    "case_id",
                    "ce_rank",
                    "ce_score",
                    "rrf_rank",
                    "rrf_score",
                    "bm25_rank",
                    "bge_rank",
                    "customer_problem",
                    "support_action",
                    "resolution_type",
                    "evidence_quality",
                    "source",
                ]
            )

        candidates = candidates.reset_index(
            drop=True
        ).copy()

        # ----------------------------------------------------
        # Build query-document pairs
        # ----------------------------------------------------

        pairs = []

        for _, row in candidates.iterrows():

            document = build_candidate_text(
                row
            )

            pairs.append(
                [
                    query,
                    document,
                ]
            )

        # ----------------------------------------------------
        # Cross-encoder prediction
        # ----------------------------------------------------

        scores = self.model.predict(
            pairs
        )

        # ----------------------------------------------------
        # Attach scores
        # ----------------------------------------------------

        candidates["ce_score"] = [
            float(score)
            for score in scores
        ]

        # ----------------------------------------------------
        # Sort by cross-encoder score
        # ----------------------------------------------------

        candidates = candidates.sort_values(
            by=[
                "ce_score",
                "case_id",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

        candidates["ce_rank"] = (
            candidates.index + 1
        )

        # ----------------------------------------------------
        # Runtime-safe output
        # ----------------------------------------------------

        output_columns = [
            "case_id",
            "ce_rank",
            "ce_score",
        ]

        optional_columns = [
            "rrf_rank",
            "rrf_score",
            "bm25_rank",
            "bge_rank",
            "customer_problem",
            "support_action",
            "resolution_type",
            "evidence_quality",
            "source",
        ]

        for column in optional_columns:

            if column in candidates.columns:
                output_columns.append(
                    column
                )

        # Explicitly prevent evaluation leakage.
        forbidden_columns = {
            "golden_id",
            "golden_intent",
            "intent_match",
            "gold_automation",
            "gold_risk",
            "gold_resolution",
        }

        output_columns = [
            column
            for column in output_columns
            if column not in forbidden_columns
        ]

        return candidates[
            output_columns
        ].head(
            top_k
        ).reset_index(
            drop=True
        )


# ============================================================
# BATCH RERANKING
# ============================================================

def batch_rerank(
    reranker: CrossEncoderReranker,
    queries: List[str],
    candidate_frames: List[pd.DataFrame],
    top_k: int = DEFAULT_OUTPUT_K,
) -> List[pd.DataFrame]:
    """
    Rerank multiple query candidate sets.
    """

    if len(queries) != len(
        candidate_frames
    ):
        raise ValueError(
            "Number of queries must match "
            "number of candidate DataFrames."
        )

    results = []

    for query, candidates in zip(
        queries,
        candidate_frames,
    ):

        results.append(
            reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        )

    return results


# ============================================================
# FAKE MODEL FOR UNIT TESTING
# ============================================================

class FakeCrossEncoder:
    """
    Deterministic fake cross-encoder used for local tests.

    It gives higher scores when the query contains a word that
    appears in the customer problem.
    """

    def predict(
        self,
        pairs,
    ):

        scores = []

        for query, document in pairs:

            query_words = set(
                str(query).lower().split()
            )

            document_words = set(
                str(document).lower().split()
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
# LOCAL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("HIVER SUPPORT AGENT — CROSS-ENCODER RERANKER TEST")
    print("=" * 100)

    # --------------------------------------------------------
    # Synthetic RRF candidates
    # --------------------------------------------------------

    candidates = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
            ],
            "rrf_rank": [
                1,
                2,
                3,
            ],
            "rrf_score": [
                0.032,
                0.031,
                0.030,
            ],
            "bm25_rank": [
                1,
                2,
                3,
            ],
            "bge_rank": [
                2,
                1,
                3,
            ],
            "customer_problem": [
                "Where is my package?",
                "I forgot my password.",
                "I need a refund.",
            ],
            "support_action": [
                "Provided package tracking guidance.",
                "Provided account recovery guidance.",
                "Explained refund options.",
            ],
            "resolution_type": [
                "ORDER_TRACKING",
                "ACCOUNT_ACCESS",
                "RETURNS_REFUNDS",
            ],
            "evidence_quality": [
                "HIGH",
                "HIGH",
                "HIGH",
            ],
            "source": [
                "historical_support_case",
                "historical_support_case",
                "historical_support_case",
            ],
        }
    )

    # --------------------------------------------------------
    # Validation test
    # --------------------------------------------------------

    validate_candidates(
        candidates
    )

    print(
        "PASS | Candidate validation"
    )

    # --------------------------------------------------------
    # Candidate text test
    # --------------------------------------------------------

    text = build_candidate_text(
        candidates.iloc[0]
    )

    assert "Where is my package?" in text
    assert "ORDER_TRACKING" in text

    print(
        "PASS | Candidate text construction"
    )

    # --------------------------------------------------------
    # Reranker initialization
    # --------------------------------------------------------

    reranker = CrossEncoderReranker(
        model=FakeCrossEncoder()
    )

    print(
        "PASS | Cross-encoder initialization"
    )

    # --------------------------------------------------------
    # Reranking test
    # --------------------------------------------------------

    results = reranker.rerank(
        query="Where is my package?",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert (
        results.iloc[0]["case_id"]
        == "CASE-001"
    )

    assert (
        results.iloc[0]["ce_rank"]
        == 1
    )

    print(
        "PASS | Cross-encoder reranking"
    )

    print()
    print("Top reranked result:")
    print(
        results.iloc[0].to_dict()
    )

    # --------------------------------------------------------
    # Top-K test
    # --------------------------------------------------------

    assert len(results) == 2

    print(
        "PASS | Top-K output"
    )

    # --------------------------------------------------------
    # Evaluation leakage test
    # --------------------------------------------------------

    candidates_with_eval_columns = (
        candidates.copy()
    )

    candidates_with_eval_columns[
        "golden_id"
    ] = "GOLD-0001"

    candidates_with_eval_columns[
        "golden_intent"
    ] = "ORDER_TRACKING"

    candidates_with_eval_columns[
        "intent_match"
    ] = True

    leakage_results = reranker.rerank(
        query="Where is my package?",
        candidates=candidates_with_eval_columns,
        top_k=3,
    )

    forbidden_columns = {
        "golden_id",
        "golden_intent",
        "intent_match",
    }

    assert not (
        forbidden_columns
        & set(leakage_results.columns)
    )

    print(
        "PASS | Evaluation-only field isolation"
    )

    # --------------------------------------------------------
    # Empty query test
    # --------------------------------------------------------

    try:

        reranker.rerank(
            query="",
            candidates=candidates,
            top_k=3,
        )

        raise AssertionError(
            "Empty query should raise ValueError."
        )

    except ValueError:
        pass

    print(
        "PASS | Empty-query validation"
    )

    # --------------------------------------------------------
    # Batch reranking test
    # --------------------------------------------------------

    batch_results = batch_rerank(
        reranker=reranker,
        queries=[
            "Where is my package?",
            "I forgot my password.",
        ],
        candidate_frames=[
            candidates,
            candidates,
        ],
        top_k=1,
    )

    assert len(batch_results) == 2

    assert (
        batch_results[0].iloc[0]["case_id"]
        == "CASE-001"
    )

    assert (
        batch_results[1].iloc[0]["case_id"]
        == "CASE-002"
    )

    print(
        "PASS | Batch reranking"
    )

    print("=" * 100)
    print("CROSS-ENCODER RERANKER TEST: PASS")
    print("=" * 100)