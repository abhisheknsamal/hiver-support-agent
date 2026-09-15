"""
BGE-large Vector Retriever
--------------------------

Semantic retrieval branch of the hybrid retrieval pipeline.

Pipeline:

    Customer Query
          ↓
    BGE-large v1.5
          ↓
    Cosine Similarity
          ↓
       Top-K Cases

The module can load a precomputed historical embedding matrix,
avoiding expensive re-embedding of the complete corpus.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# OPTIONAL DEPENDENCY
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

DEFAULT_EMBEDDING_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "embeddings"
    / "historical_bge_large.npy"
)


# ============================================================
# VECTOR RETRIEVER
# ============================================================

class BGEVectorRetriever:
    """
    Semantic retriever using BGE-large embeddings.
    """

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        embedding_path: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model=None,
    ):
        """
        Initialize the vector retriever.

        Args:
            corpus_df:
                Historical structured resolution corpus.

            embedding_path:
                Path to precomputed corpus embeddings.

            model_name:
                SentenceTransformer model name.

            model:
                Optional preloaded model, useful for testing.
        """

        if not isinstance(corpus_df, pd.DataFrame):
            raise TypeError(
                "corpus_df must be a pandas DataFrame."
            )

        if "case_id" not in corpus_df.columns:
            raise ValueError(
                "Corpus must contain case_id."
            )

        if "retrieval_text" not in corpus_df.columns:
            raise ValueError(
                "Corpus must contain retrieval_text."
            )

        self.corpus_df = corpus_df.reset_index(
            drop=True
        ).copy()

        self.model_name = model_name

        # ----------------------------------------------------
        # Load precomputed corpus embeddings
        # ----------------------------------------------------

        if embedding_path is None:
            embedding_path = DEFAULT_EMBEDDING_PATH

        self.embedding_path = Path(
            embedding_path
        )

        if not self.embedding_path.exists():
            raise FileNotFoundError(
                f"Embedding matrix not found: "
                f"{self.embedding_path}\n\n"
                "Provide the path to the previously generated "
                "BGE-large embedding matrix."
            )

        self.embeddings = np.load(
            self.embedding_path
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Embedding matrix must be 2-dimensional."
            )

        if len(self.embeddings) != len(
            self.corpus_df
        ):
            raise ValueError(
                "Embedding count does not match corpus size: "
                f"{len(self.embeddings):,} embeddings vs "
                f"{len(self.corpus_df):,} corpus rows."
            )

        # ----------------------------------------------------
        # Normalize corpus embeddings
        # ----------------------------------------------------

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        self.normalized_embeddings = (
            self.embeddings / norms
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if model is not None:
            self.model = model

        else:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed.\n"
                    "Install it with:\n"
                    "pip install sentence-transformers"
                )

            self.model = SentenceTransformer(
                model_name
            )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Encode and L2-normalize a query.
        """

        if not isinstance(query, str):
            query = str(query)

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)

        return embedding

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Retrieve top-K historical cases using cosine similarity.

        Returns:

            case_id
            bge_rank
            bge_score
            customer_problem
            support_action
            resolution_type
            evidence_quality
            source
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = self.encode_query(
            query
        )

        # ----------------------------------------------------
        # Cosine similarity because both vectors are normalized
        # ----------------------------------------------------

        scores = (
            self.normalized_embeddings
            @ query_embedding
        )

        top_k = min(
            top_k,
            len(scores),
        )

        ranked_indices = np.argsort(
            -scores
        )[:top_k]

        rows = []

        for rank, idx in enumerate(
            ranked_indices,
            start=1,
        ):

            row = self.corpus_df.iloc[
                int(idx)
            ]

            rows.append(
                {
                    "case_id": row["case_id"],
                    "bge_rank": rank,
                    "bge_score": float(
                        scores[idx]
                    ),
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


# ============================================================
# LOCAL TEST MODEL
# ============================================================

class FakeEmbeddingModel:
    """
    Tiny deterministic embedding model for unit testing.

    This avoids downloading BGE-large during repository tests.
    """

    def encode(
        self,
        query,
        normalize_embeddings=True,
    ):
        mapping = {
            "package": np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            "password": np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
            "refund": np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        }

        text = str(query).lower()

        if "package" in text:
            vector = mapping["package"]

        elif "password" in text:
            vector = mapping["password"]

        else:
            vector = mapping["refund"]

        return vector


# ============================================================
# LOCAL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("HIVER SUPPORT AGENT — BGE VECTOR RETRIEVER TEST")
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
            ],
            "customer_problem": [
                "Where is my package?",
                "I forgot my password.",
                "I need a refund.",
            ],
            "support_action": [
                "Provided delivery tracking guidance.",
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

    test_df["retrieval_text"] = (
        "Customer problem: "
        + test_df["customer_problem"]
        + "\nSupport action: "
        + test_df["support_action"]
        + "\nResolution type: "
        + test_df["resolution_type"]
    )

    # --------------------------------------------------------
    # Synthetic normalized embeddings
    # --------------------------------------------------------

    test_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    temp_embedding_path = (
        Path(__file__).resolve().parent
        / "_test_embeddings.npy"
    )

    np.save(
        temp_embedding_path,
        test_embeddings,
    )

    try:

        # ----------------------------------------------------
        # Initialize retriever
        # ----------------------------------------------------

        retriever = BGEVectorRetriever(
            corpus_df=test_df,
            embedding_path=str(
                temp_embedding_path
            ),
            model=FakeEmbeddingModel(),
        )

        print(
            "PASS | Embedding matrix loading"
        )

        print(
            "PASS | Embedding/corpus size validation"
        )

        print(
            "PASS | Embedding normalization"
        )

        # ----------------------------------------------------
        # Search test
        # ----------------------------------------------------

        results = retriever.search(
            query="Where is my package?",
            top_k=2,
        )

        assert len(results) == 2

        assert (
            results.iloc[0]["case_id"]
            == "CASE-001"
        )

        assert (
            results.iloc[0]["bge_rank"]
            == 1
        )

        assert (
            results.iloc[0]["bge_score"]
            > 0.99
        )

        print(
            "PASS | Top-K semantic retrieval"
        )

        print()
        print("Top BGE result:")
        print(
            results.iloc[0].to_dict()
        )

        # ----------------------------------------------------
        # Second query
        # ----------------------------------------------------

        password_results = retriever.search(
            query="I forgot my password",
            top_k=1,
        )

        assert (
            password_results.iloc[0]["case_id"]
            == "CASE-002"
        )

        print(
            "PASS | Multiple semantic queries"
        )

        # ----------------------------------------------------
        # Empty query
        # ----------------------------------------------------

        try:
            retriever.search(
                query="",
                top_k=5,
            )

            raise AssertionError(
                "Empty query should raise ValueError."
            )

        except ValueError:
            pass

        print(
            "PASS | Empty-query validation"
        )

    finally:

        if temp_embedding_path.exists():
            temp_embedding_path.unlink()

    print("=" * 100)
    print("BGE VECTOR RETRIEVER TEST: PASS")
    print("=" * 100)