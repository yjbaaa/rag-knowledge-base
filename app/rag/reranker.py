"""
Reranker - Cross-Encoder based document re-ranking
Refines the top-k retrieval results for better precision.
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from app.core.config import RERANKER_MODEL, ENABLE_RERANKER


class Reranker:
    """
    Cross-Encoder reranker using sentence-transformers.
    Scores (query, document) pairs for fine-grained relevance ranking.

    Falls back to identity (no-op) if sentence-transformers is not installed
    or ENABLE_RERANKER is False.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
            )
        except Exception as e:
            print(f"[WARN] Reranker model not available: {e}")
            print(f"[INFO] Install with: pip install sentence-transformers")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Re-rank documents by Cross-Encoder relevance scores.

        Args:
            query: The search query.
            documents: Candidate documents from retrieval.
            top_k: Number of top documents to return (None = return all).

        Returns:
            Re-ranked document list.
        """
        if not ENABLE_RERANKER or not documents:
            return documents

        self._load_model()

        if self._model is None:
            return documents  # No-op fallback

        if len(documents) <= 1:
            return documents

        # Score each (query, document) pair
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        # Sort by score descending
        scored = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_k = top_k or len(documents)
        return [doc for doc, _ in scored[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank and return (document, score) tuples.
        Score range: approximately 0-1 (higher is better).
        """
        if not ENABLE_RERANKER or not documents:
            return [(doc, 0.0) for doc in documents]

        self._load_model()

        if self._model is None:
            return [(doc, 0.0) for doc in documents]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)

        # Normalize to [0, 1] using sigmoid
        import math
        normalized = [1.0 / (1.0 + math.exp(-s)) for s in scores]

        scored = sorted(
            zip(documents, normalized),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored


# ========== Global singleton ==========

_reranker_instance: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """Convenience: re-rank documents."""
    return get_reranker().rerank(query, documents, top_k)
