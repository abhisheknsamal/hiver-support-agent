"""
BM25 Historical Evidence Retriever
-----------------------------------

Retrieves historical customer-support cases using BM25 lexical
matching.

Pipeline position:

    Customer Query
          ↓
        BM25
          ↓
      Top-K Cases

The retriever operates on the structured case-level corpus created
by corpus.py.

It does NOT use the 200-case golden evaluation set as runtime data.
"""

from typing import Dict, List, Optional

import re

import pandas as pd


# ============================================================
# OPTIONAL DEPENDENCY
# ============================================================

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25.

    Uses a simple lowercase alphanumeric tokenizer so that the same
    normalization can be applied to both corpus documents and queries.
    """

    if text is None:
        return []

    text = str(text).lower()

    return re.findall(
        r"\b[a-z0-9]+\b",
        text,
    )


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:
    """
    BM25 retriever over a structured historical support corpus.
    """

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        text_column: str = "retrieval_text",
    ):
        """
        Initialize the BM25 index.

        Args:
            corpus_df:
                Historical resolution corpus.

            text_column:
                Column containing retrieval text.
        """

        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25 is not installed.\n"
                "Install it with:\n"
                "pip install rank-bm25"
            )

        if not isinstance(corpus_df, pd.DataFrame):
            raise TypeError(
                "corpus_df must be a pandas DataFrame."
            )

        if text_column not in corpus_df.columns:
            raise ValueError(
                f"Missing text column: {text_column}"
            )

        if "case_id" not in corpus_df.columns:
            raise ValueError(
                "Corpus must contain a case_id column."
            )

        self.corpus_df = corpus_df.reset_index(
            drop=True
        ).copy()

        self.text_column = text_column

        self.tokenized_corpus = [
            tokenize(text)
            for text in self.corpus_df[text_column]
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Retrieve the top-K historical cases.

        Returns columns:

            case_id
            bm25_rank
            bm25_score
            customer_problem
            support_action
            resolution_type
            evidence_quality
            source
        """

        if not isinstance(query, str):
            query = str(query)

        query = query.strip()

        if not query:
            return pd.DataFrame(
                columns=[
                    "case_id",
                    "bm25_rank",
                    "bm25_score",
                    "customer_problem",
                    "support_action",
                    "resolution_type",
                    "evidence_quality",
                    "source",
                ]
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_tokens = tokenize(query)

        if not query_tokens:
            return pd.DataFrame(
                columns=[
                    "case_id",
                    "bm25_rank",
                    "bm25_score",
                    "customer_problem",
                    "support_action",
                    "resolution_type",
                    "evidence_quality",
                    "source",
                ]
            )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        rows = []

        for rank, idx in enumerate(
            ranked_indices,
            start=1,
        ):

            row = self.corpus_df.iloc[idx]

            rows.append(
                {
                    "case_id": row["case_id"],
                    "bm25_rank": rank,
                    "bm25_score": float(scores[idx]),
                    "customer_problem": row[
                        "customer_problem"
                    ],
                    "support_action": row[
                        "support_action"
                    ],
                    "resolution_type": row[
                        "resolution_type"
                    ],
                    "evidence_quality": row[
                        "evidence_quality"
                    ],
                    "source": row["source"],
                }
            )

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Batch Search
    # --------------------------------------------------------

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical evidence for multiple queries.
        """

        results = {}

        for query in queries:
            results[query] = self.search(
                query=query,
                top_k=top_k,
            )

        return results


# ============================================================
# LOCAL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("HIVER SUPPORT AGENT — BM25 RETRIEVER TEST")
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
                "My order arrived damaged.",
                "I forgot my password.",
                "I was charged twice.",
            ],
            "support_action": [
                "Provided delivery tracking guidance.",
                "Explained return and refund options.",
                "Provided account recovery guidance.",
                "Escalated the duplicate payment issue.",
            ],
            "resolution_type": [
                "ORDER_TRACKING",
                "RETURNS_REFUNDS",
                "ACCOUNT_ACCESS",
                "PAYMENT_BILLING",
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

    # Build retrieval text locally so the test does not depend on
    # the external corpus file.
    test_df["retrieval_text"] = (
        "Customer problem: "
        + test_df["customer_problem"]
        + "\nSupport action: "
        + test_df["support_action"]
        + "\nResolution type: "
        + test_df["resolution_type"]
    )

    # --------------------------------------------------------
    # Tokenization test
    # --------------------------------------------------------

    tokens = tokenize(
        "Where is my package?"
    )

    assert tokens == [
        "where",
        "is",
        "my",
        "package",
    ]

    print(
        "PASS | Tokenization"
    )

    # --------------------------------------------------------
    # Retriever initialization
    # --------------------------------------------------------

    retriever = BM25Retriever(
        corpus_df=test_df
    )

    print(
        "PASS | BM25 index construction"
    )

    # --------------------------------------------------------
    # Search test
    # --------------------------------------------------------

    results = retriever.search(
        query="Where is my package?",
        top_k=3,
    )

    assert len(results) == 3

    assert results.iloc[0]["case_id"] == "CASE-001"

    assert (
        results.iloc[0]["bm25_rank"]
        == 1
    )

    print(
        "PASS | Top-K BM25 retrieval"
    )

    print()
    print("Top BM25 result:")
    print(
        results.iloc[0].to_dict()
    )

    # --------------------------------------------------------
    # Empty query test
    # --------------------------------------------------------

    empty_results = retriever.search(
        query="",
        top_k=5,
    )

    assert empty_results.empty

    print(
        "PASS | Empty-query handling"
    )

    # --------------------------------------------------------
    # Batch search test
    # --------------------------------------------------------

    batch_results = retriever.batch_search(
        queries=[
            "Where is my package?",
            "I forgot my password",
        ],
        top_k=2,
    )

    assert len(batch_results) == 2

    assert (
        batch_results[
            "Where is my package?"
        ].iloc[0]["case_id"]
        == "CASE-001"
    )

    assert (
        batch_results[
            "I forgot my password"
        ].iloc[0]["case_id"]
        == "CASE-003"
    )

    print(
        "PASS | Batch retrieval"
    )

    print("=" * 100)
    print("BM25 RETRIEVER TEST: PASS")
    print("=" * 100)