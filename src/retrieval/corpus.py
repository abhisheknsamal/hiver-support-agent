"""
Historical Resolution Corpus
----------------------------

Loads the structured historical customer-support resolution corpus
used by the retrieval pipeline.

Expected corpus columns:

    case_id
    customer_problem
    support_action
    resolution_type
    evidence_quality
    source

Important:
-----------
The 200-case golden evaluation set must remain excluded from the
runtime retrieval corpus.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
)

DEFAULT_CORPUS_PATH = (
    DEFAULT_DATA_DIR
    / "evaluation"
    / "structured_rag_corpus.csv"
)

REQUIRED_COLUMNS = {
    "case_id",
    "customer_problem",
    "support_action",
    "resolution_type",
    "evidence_quality",
    "source",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_corpus_schema(
    df: pd.DataFrame,
) -> None:
    """
    Validate that the corpus contains the required columns.

    Raises:
        ValueError: if required columns are missing.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Historical corpus is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


# ============================================================
# TEXT CONSTRUCTION
# ============================================================

def build_retrieval_text(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the text representation used by retrieval systems.

    The retrieval text combines:

        customer_problem
        support_action
        resolution_type

    This keeps the retrieval object at the case/resolution level
    rather than indexing raw tweets.
    """

    result = df.copy()

    result["customer_problem"] = (
        result["customer_problem"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["support_action"] = (
        result["support_action"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["resolution_type"] = (
        result["resolution_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["retrieval_text"] = (
        "Customer problem: "
        + result["customer_problem"]
        + "\nSupport action: "
        + result["support_action"]
        + "\nResolution type: "
        + result["resolution_type"]
    )

    return result


# ============================================================
# CORPUS LOADER
# ============================================================

def load_corpus(
    corpus_path: Optional[str] = None,
    expected_min_rows: Optional[int] = 80000,
) -> pd.DataFrame:
    """
    Load and validate the structured historical corpus.

    Args:
        corpus_path:
            Path to structured_rag_corpus.csv.

        expected_min_rows:
            Optional sanity check. The completed historical
            corpus contains approximately 83k cases.

    Returns:
        Validated DataFrame with retrieval_text added.
    """

    path = (
        Path(corpus_path)
        if corpus_path is not None
        else DEFAULT_CORPUS_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Historical corpus not found: {path}\n\n"
            "The repository intentionally does not contain the "
            "large runtime data artifact. Provide the corpus path "
            "explicitly when running the retrieval pipeline."
        )

    df = pd.read_csv(path)

    validate_corpus_schema(df)

    if expected_min_rows is not None:
        if len(df) < expected_min_rows:
            raise ValueError(
                f"Corpus contains only {len(df):,} rows. "
                f"Expected at least {expected_min_rows:,} rows."
            )

    if df["case_id"].isna().any():
        raise ValueError(
            "Corpus contains missing case_id values."
        )

    duplicate_case_ids = (
        df["case_id"].duplicated().sum()
    )

    if duplicate_case_ids > 0:
        raise ValueError(
            f"Corpus contains {duplicate_case_ids:,} "
            "duplicate case_id values."
        )

    df = build_retrieval_text(df)

    return df


# ============================================================
# CORPUS SUMMARY
# ============================================================

def corpus_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Return useful corpus statistics.
    """

    validate_corpus_schema(df)

    return {
        "row_count": int(len(df)),
        "unique_case_ids": int(
            df["case_id"].nunique()
        ),
        "resolution_types": (
            df["resolution_type"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "evidence_quality": (
            df["evidence_quality"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "sources": (
            df["source"]
            .value_counts(dropna=False)
            .to_dict()
        ),
    }


# ============================================================
# RUNTIME CORPUS FILTER
# ============================================================

def exclude_evaluation_cases(
    df: pd.DataFrame,
    evaluation_case_ids: Optional[set] = None,
) -> pd.DataFrame:
    """
    Explicitly remove evaluation cases from the runtime corpus.

    This function is intentionally conservative.

    If evaluation_case_ids is provided, those cases are removed.

    If it is not provided, the corpus is returned unchanged because
    we must not guess which records belong to the evaluation set.
    """

    if evaluation_case_ids is None:
        return df.copy()

    if not evaluation_case_ids:
        return df.copy()

    filtered = df[
        ~df["case_id"].isin(evaluation_case_ids)
    ].copy()

    return filtered.reset_index(drop=True)


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("HIVER SUPPORT AGENT — HISTORICAL CORPUS MODULE TEST")
    print("=" * 100)

    # --------------------------------------------------------
    # Schema test using a small synthetic corpus
    # --------------------------------------------------------

    test_df = pd.DataFrame(
        {
            "case_id": ["CASE-001", "CASE-002"],
            "customer_problem": [
                "Where is my order?",
                "I received a damaged item.",
            ],
            "support_action": [
                "Provided delivery tracking guidance.",
                "Explained the return/refund process.",
            ],
            "resolution_type": [
                "ORDER_TRACKING",
                "RETURNS_REFUNDS",
            ],
            "evidence_quality": [
                "HIGH",
                "HIGH",
            ],
            "source": [
                "historical_support_case",
                "historical_support_case",
            ],
        }
    )

    validate_corpus_schema(test_df)

    test_df = build_retrieval_text(test_df)

    assert "retrieval_text" in test_df.columns
    assert len(test_df) == 2
    assert test_df["case_id"].nunique() == 2

    print("PASS | Required schema validation")
    print("PASS | Retrieval text construction")
    print("PASS | Case ID uniqueness")

    # --------------------------------------------------------
    # Evaluation exclusion test
    # --------------------------------------------------------

    evaluation_ids = {
        "CASE-001"
    }

    filtered_df = exclude_evaluation_cases(
        test_df,
        evaluation_case_ids=evaluation_ids,
    )

    assert len(filtered_df) == 1
    assert "CASE-001" not in set(
        filtered_df["case_id"]
    )

    print("PASS | Evaluation-case exclusion")

    # --------------------------------------------------------
    # Summary test
    # --------------------------------------------------------

    summary = corpus_summary(test_df)

    assert summary["row_count"] == 2
    assert summary["unique_case_ids"] == 2

    print("PASS | Corpus summary")

    print("=" * 100)
    print("CORPUS MODULE TEST: PASS")
    print("=" * 100)