"""
Reciprocal Rank Fusion (RRF)
----------------------------

Combines independent retrieval rankings from:

    BM25
    BGE-large semantic retrieval

into one unified ranking.

Formula:

    RRF(d) = Σ 1 / (k + rank(d))

The default k=60 matches the retrieval configuration used in
the Hiver Support Agent evaluation pipeline.

Runtime output intentionally contains only retrieval-safe fields.
Evaluation-only fields such as golden_intent and intent_match
must never enter the runtime fusion result.
"""

from typing import Dict, List

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RRF_K = 60


# ============================================================
# VALIDATION
# ============================================================

def validate_retrieval_frame(
    df: pd.DataFrame,
    rank_column: str,
) -> None:
    """
    Validate a retrieval result DataFrame.
    """

    required_columns = {
        "case_id",
        rank_column,
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    if df["case_id"].duplicated().any():
        raise ValueError(
            "Retrieval results contain duplicate case_id values."
        )


# ============================================================
# RRF SCORE
# ============================================================

def calculate_rrf_score(
    rank: int,
    k: int = DEFAULT_RRF_K,
) -> float:
    """
    Calculate the RRF contribution for one ranking position.

        score = 1 / (k + rank)
    """

    if rank <= 0:
        raise ValueError(
            "Rank must be greater than zero."
        )

    if k < 0:
        raise ValueError(
            "RRF k must be non-negative."
        )

    return 1.0 / (
        k + rank
    )


# ============================================================
# RRF FUSION
# ============================================================

def reciprocal_rank_fusion(
    bm25_results: pd.DataFrame,
    bge_results: pd.DataFrame,
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Fuse BM25 and BGE retrieval rankings.

    Args:
        bm25_results:
            BM25 retrieval DataFrame containing:
                case_id
                bm25_rank

        bge_results:
            BGE retrieval DataFrame containing:
                case_id
                bge_rank

        k:
            RRF smoothing constant. Default = 60.

        top_k:
            Number of fused results to return.

    Returns:
        DataFrame containing:

            case_id
            rrf_rank
            rrf_score
            bm25_rank
            bge_rank
            customer_problem
            support_action
            resolution_type
            evidence_quality
            source

    A case appearing in both rankings receives contributions
    from both retrieval systems.
    """

    if k < 0:
        raise ValueError(
            "RRF k must be non-negative."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    validate_retrieval_frame(
        bm25_results,
        "bm25_rank",
    )

    validate_retrieval_frame(
        bge_results,
        "bge_rank",
    )

    # --------------------------------------------------------
    # Store retrieval metadata
    # --------------------------------------------------------

    bm25_lookup = {}

    for _, row in bm25_results.iterrows():
        bm25_lookup[row["case_id"]] = {
            "rank": int(row["bm25_rank"]),
            "score": calculate_rrf_score(
                int(row["bm25_rank"]),
                k,
            ),
        }

    bge_lookup = {}

    for _, row in bge_results.iterrows():
        bge_lookup[row["case_id"]] = {
            "rank": int(row["bge_rank"]),
            "score": calculate_rrf_score(
                int(row["bge_rank"]),
                k,
            ),
        }

    # --------------------------------------------------------
    # Union of candidate cases
    # --------------------------------------------------------

    candidate_ids = (
        set(bm25_lookup.keys())
        | set(bge_lookup.keys())
    )

    fused_rows = []

    for case_id in candidate_ids:

        bm25_info = bm25_lookup.get(
            case_id
        )

        bge_info = bge_lookup.get(
            case_id
        )

        bm25_score = (
            bm25_info["score"]
            if bm25_info is not None
            else 0.0
        )

        bge_score = (
            bge_info["score"]
            if bge_info is not None
            else 0.0
        )

        rrf_score = (
            bm25_score
            + bge_score
        )

        fused_rows.append(
            {
                "case_id": case_id,
                "rrf_score": float(
                    rrf_score
                ),
                "bm25_rank": (
                    bm25_info["rank"]
                    if bm25_info is not None
                    else None
                ),
                "bge_rank": (
                    bge_info["rank"]
                    if bge_info is not None
                    else None
                ),
            }
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    fused_df = pd.DataFrame(
        fused_rows
    )

    fused_df = fused_df.sort_values(
        by=[
            "rrf_score",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Add RRF rank
    # --------------------------------------------------------

    fused_df["rrf_rank"] = (
        fused_df.index + 1
    )

    # --------------------------------------------------------
    # Attach case metadata
    # --------------------------------------------------------

    metadata_columns = [
        "case_id",
        "customer_problem",
        "support_action",
        "resolution_type",
        "evidence_quality",
        "source",
    ]

    metadata_source = None

    if all(
        column in bm25_results.columns
        for column in metadata_columns
    ):
        metadata_source = bm25_results[
            metadata_columns
        ].copy()

    elif all(
        column in bge_results.columns
        for column in metadata_columns
    ):
        metadata_source = bge_results[
            metadata_columns
        ].copy()

    if metadata_source is not None:

        fused_df = fused_df.merge(
            metadata_source,
            on="case_id",
            how="left",
        )

    # --------------------------------------------------------
    # Runtime-safe column order
    # --------------------------------------------------------

    output_columns = [
        "case_id",
        "rrf_rank",
        "rrf_score",
        "bm25_rank",
        "bge_rank",
    ]

    metadata_output = [
        "customer_problem",
        "support_action",
        "resolution_type",
        "evidence_quality",
        "source",
    ]

    for column in metadata_output:

        if column in fused_df.columns:
            output_columns.append(
                column
            )

    fused_df = fused_df[
        output_columns
    ]

    return fused_df.head(
        top_k
    ).reset_index(
        drop=True
    )


# ============================================================
# BATCH FUSION
# ============================================================

def batch_reciprocal_rank_fusion(
    bm25_results_list: List[pd.DataFrame],
    bge_results_list: List[pd.DataFrame],
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> List[pd.DataFrame]:
    """
    Fuse corresponding BM25 and BGE result sets.
    """

    if len(bm25_results_list) != len(
        bge_results_list
    ):
        raise ValueError(
            "BM25 and BGE result lists must "
            "have the same length."
        )

    results = []

    for bm25_results, bge_results in zip(
        bm25_results_list,
        bge_results_list,
    ):

        results.append(
            reciprocal_rank_fusion(
                bm25_results=bm25_results,
                bge_results=bge_results,
                k=k,
                top_k=top_k,
            )
        )

    return results


# ============================================================
# LOCAL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("HIVER SUPPORT AGENT — RRF FUSION TEST")
    print("=" * 100)

    # --------------------------------------------------------
    # Synthetic BM25 results
    # --------------------------------------------------------

    bm25_results = pd.DataFrame(
        {
            "case_id": [
                "CASE-A",
                "CASE-B",
                "CASE-C",
            ],
            "bm25_rank": [
                1,
                2,
                3,
            ],
            "bm25_score": [
                5.0,
                4.0,
                3.0,
            ],
            "customer_problem": [
                "Package is delayed.",
                "Password forgotten.",
                "Need a refund.",
            ],
            "support_action": [
                "Provided delivery guidance.",
                "Provided account recovery guidance.",
                "Explained refund process.",
            ],
            "resolution_type": [
                "ORDER_DELIVERY",
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
    # Synthetic BGE results
    # --------------------------------------------------------

    bge_results = pd.DataFrame(
        {
            "case_id": [
                "CASE-B",
                "CASE-A",
                "CASE-D",
            ],
            "bge_rank": [
                1,
                2,
                3,
            ],
            "bge_score": [
                0.95,
                0.90,
                0.80,
            ],
            "customer_problem": [
                "Password forgotten.",
                "Package is delayed.",
                "Payment issue.",
            ],
            "support_action": [
                "Provided account recovery guidance.",
                "Provided delivery guidance.",
                "Investigated payment issue.",
            ],
            "resolution_type": [
                "ACCOUNT_ACCESS",
                "ORDER_DELIVERY",
                "PAYMENT_BILLING",
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
    # RRF formula test
    # --------------------------------------------------------

    score = calculate_rrf_score(
        rank=1,
        k=60,
    )

    assert abs(
        score - (1 / 61)
    ) < 1e-10

    print(
        "PASS | RRF score calculation"
    )

    # --------------------------------------------------------
    # Fusion test
    # --------------------------------------------------------

    results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        bge_results=bge_results,
        k=60,
        top_k=4,
    )

    assert len(results) == 4

    print(
        "PASS | BM25 + BGE candidate fusion"
    )

    # CASE-A:
    # BM25 rank 1 + BGE rank 2
    case_a = results[
        results["case_id"] == "CASE-A"
    ].iloc[0]

    expected_case_a = (
        (1 / 61)
        + (1 / 62)
    )

    assert abs(
        case_a["rrf_score"]
        - expected_case_a
    ) < 1e-10

    print(
        "PASS | Correct RRF score for shared case"
    )

    # CASE-D appears only in BGE.
    case_d = results[
        results["case_id"] == "CASE-D"
    ].iloc[0]

    expected_case_d = (
        1 / 63
    )

    assert abs(
        case_d["rrf_score"]
        - expected_case_d
    ) < 1e-10

    print(
        "PASS | Single-retriever candidate handling"
    )

    # --------------------------------------------------------
    # Ranking test
    # --------------------------------------------------------

    assert (
        results.iloc[0]["rrf_score"]
        >= results.iloc[1]["rrf_score"]
    )

    print(
        "PASS | Unified ranking order"
    )

    # --------------------------------------------------------
    # Runtime safety test
    # --------------------------------------------------------

    forbidden_columns = {
        "golden_id",
        "golden_intent",
        "intent_match",
    }

    assert not (
        forbidden_columns
        & set(results.columns)
    )

    print(
        "PASS | Evaluation-only field isolation"
    )

    # --------------------------------------------------------
    # Batch fusion test
    # --------------------------------------------------------

    batch_results = (
        batch_reciprocal_rank_fusion(
            bm25_results_list=[
                bm25_results
            ],
            bge_results_list=[
                bge_results
            ],
            top_k=3,
        )
    )

    assert len(batch_results) == 1
    assert len(batch_results[0]) == 3

    print(
        "PASS | Batch RRF fusion"
    )

    print()
    print("Top fused result:")
    print(
        results.iloc[0].to_dict()
    )

    print("=" * 100)
    print("RRF FUSION TEST: PASS")
    print("=" * 100)