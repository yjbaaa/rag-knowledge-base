"""
Hybrid Search Engine
BM25 keyword + Vector semantic + RRF fusion ranking
"""

from typing import List, Optional, Tuple

import time
import math
import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from app.core.config import (
    CHROMA_COLLECTION_NAME,
    DB_DIR,
    RETRIEVAL_TOP_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
)
from app.rag.embedding import get_embedding


# =====================================================================
#  BM25 Keyword Search Engine (Incremental)
# =====================================================================

# Pre-compiled tokenization regex (avoid re-compile on every call)
_TOKEN_SPLIT_RE = re.compile(r"(\s+)")
_TOKEN_EXTRACT_RE = re.compile(r"[一-鿿]|[a-zA-Z]+|\d+|[^\s\w]")

_TOKENIZE_TEXT_CACHE = {}


class BM25Engine:
    """
    Incremental BM25 keyword search engine.

    Avoids full-corpus rebuild on every index() call by maintaining
    running document-frequency and per-document term-frequency stats.
    New docs only update their own stats + global DF counts.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Document] = []
        self._ids: set = set()  # chunk_id dedup
        # Corpus-level stats
        self._N: int = 0
        self._sum_doc_len: float = 0.0
        # term -> document frequency (number of docs containing the term)
        self._df: dict = {}
        # Per-document records: list of (doc_length, term_freq_dict)
        self._doc_records: list = []

    def index(self, documents: List[Document]):
        """Incrementally add documents without full rebuild."""
        import time
        t0 = time.perf_counter()
        new_docs = []
        for doc in documents:
            cid = doc.metadata.get("chunk_id")
            if cid and cid in self._ids:
                continue
            new_docs.append(doc)
            if cid:
                self._ids.add(cid)
        t_dedup = time.perf_counter() - t0

        if not new_docs:
            print(f"[INDEX-TIMING] bm25.index: {t_dedup:.3f}s (new=0, skipped={len(documents)} dup) "
                  f"| total_corpus={self._N}")
            return

        # Tokenize only new docs + update stats incrementally
        t1 = time.perf_counter()
        for doc in new_docs:
            tokens = self._tokenize(doc.page_content)
            doc_len = len(tokens)
            term_freq = {}
            for t in tokens:
                term_freq[t] = term_freq.get(t, 0) + 1

            self._docs.append(doc)
            self._doc_records.append((doc_len, term_freq))
            self._N += 1
            self._sum_doc_len += doc_len

            # Update document frequencies (only for terms in this doc)
            for term in term_freq:
                self._df[term] = self._df.get(term, 0) + 1
        t_tok = time.perf_counter() - t1

        t_total = time.perf_counter() - t0
        print(f"[INDEX-TIMING] bm25.index: {t_total:.3f}s "
              f"(new={len(new_docs)} skipped={len(documents)-len(new_docs)} dup) | "
              f"dedup={t_dedup:.3f}s tokenize+update={t_tok:.3f}s "
              f"[FULL CORPUS={self._N}]")

    @property
    def size(self) -> int:
        return self._N

    @property
    def _avgdl(self) -> float:
        return self._sum_doc_len / self._N if self._N else 0.0

    @staticmethod
    def _idf(df: int, N: int) -> float:
        """BM25 IDF formula."""
        return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Tuple[Document, float]]:
        if self._N == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        avgdl = self._avgdl
        k1 = self.k1
        b = self.b

        # Pre-compute IDF for each unique query term
        unique_q_terms = set(query_tokens)
        term_idf = {}
        for term in unique_q_terms:
            df = self._df.get(term, 0)
            if df > 0:
                term_idf[term] = self._idf(df, self._N)

        if not term_idf:
            return []

        # Score each document
        scores = []
        for idx, (doc_len, term_freq) in enumerate(self._doc_records):
            score = 0.0
            norm = 1.0 - b + b * (doc_len / avgdl)
            for term in unique_q_terms:
                if term not in term_idf:
                    continue
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                numerator = term_idf[term] * tf * (k1 + 1.0)
                denominator = tf + k1 * norm
                score += numerator / denominator
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        max_score = top[0][1] if top else 1.0
        return [
            (self._docs[idx], score / max_score)
            for idx, score in top
        ]

    def sync_from_docs(self, documents: list):
        """Bulk-load BM25 from existing documents (e.g. ChromaDB recovery)."""
        import time
        t0 = time.perf_counter()
        new_count = 0
        for doc in documents:
            cid = doc.metadata.get("chunk_id")
            if cid and cid in self._ids:
                continue
            if cid:
                self._ids.add(cid)
            tokens = self._tokenize(doc.page_content)
            doc_len = len(tokens)
            term_freq = {}
            for t in tokens:
                term_freq[t] = term_freq.get(t, 0) + 1
            self._docs.append(doc)
            self._doc_records.append((doc_len, term_freq))
            self._N += 1
            self._sum_doc_len += doc_len
            for term in term_freq:
                self._df[term] = self._df.get(term, 0) + 1
            new_count += 1
        t_total = time.perf_counter() - t0
        if new_count:
            print(f"[INDEX-TIMING] bm25.sync: {t_total:.3f}s (loaded={new_count}, total={self._N})")

    def delete_by_source(self, source_path: str) -> int:
        """Delete all documents matching source_path. Returns number of docs deleted."""
        import time
        t0 = time.perf_counter()
        indices_to_remove = []
        for idx, doc in enumerate(self._docs):
            if doc.metadata.get("source") == source_path:
                indices_to_remove.append(idx)
        if not indices_to_remove:
            print(f"[INDEX-TIMING] bm25.delete: 0.000s (deleted=0)")
            return 0
        for idx in reversed(indices_to_remove):
            doc = self._docs[idx]
            doc_len, term_freq = self._doc_records[idx]
            for term in term_freq:
                self._df[term] = self._df.get(term, 1) - 1
                if self._df[term] <= 0:
                    del self._df[term]
            self._N -= 1
            self._sum_doc_len -= doc_len
            cid = doc.metadata.get("chunk_id")
            if cid and cid in self._ids:
                self._ids.discard(cid)
            del self._docs[idx]
            del self._doc_records[idx]
        t_total = time.perf_counter() - t0
        deleted = len(indices_to_remove)
        print(f"[INDEX-TIMING] bm25.delete: {t_total:.3f}s (deleted={deleted}, remaining={self._N})")
        return deleted

    def clear_all(self):
        self._docs.clear()
        self._ids.clear()
        self._doc_records.clear()
        self._df.clear()
        self._N = 0
        self._sum_doc_len = 0.0

    def get_indexed_sources(self) -> list:
        """Return list of unique source paths currently indexed."""
        seen = set()
        sources = []
        for doc in self._docs:
            src = doc.metadata.get("source", "")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25 (Chinese-aware), with pre-compiled regex."""
        tokens = []
        for part in _TOKEN_SPLIT_RE.split(text):
            if part.strip():
                sub_parts = _TOKEN_EXTRACT_RE.findall(part)
                tokens.extend(t.lower() for t in sub_parts if t.strip())
        return tokens


# =====================================================================
#  Vector Semantic Search Engine
# =====================================================================

class VectorEngine:
    """ChromaDB-based vector semantic search engine."""

    def __init__(self):
        import time
        t0 = time.perf_counter()
        self._client = chromadb.PersistentClient(
            path=str(DB_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        t_client = time.perf_counter() - t0

        t1 = time.perf_counter()
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        t_collection = time.perf_counter() - t1

        t2 = time.perf_counter()
        self._embedding = get_embedding()
        t_embed_load = time.perf_counter() - t2

        self._docs: List[Document] = []
        print(f"[INDEX-TIMING] vector.__init__: {t_client+t_collection+t_embed_load:.3f}s | "
              f"chroma_client={t_client:.3f}s get_collection={t_collection:.3f}s "
              f"load_embedding_model={t_embed_load:.3f}s")

    def index(self, documents: List[Document]):
        if not documents:
            return
        import time
        t0 = time.perf_counter()

        texts = [doc.page_content for doc in documents]
        metadatas = [dict(doc.metadata) for doc in documents]
        ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]

        t_prep = time.perf_counter() - t0

        t1 = time.perf_counter()
        embeddings = self._embedding.embed_documents(texts)
        t_embed = time.perf_counter() - t1

        t2 = time.perf_counter()
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        t_upsert = time.perf_counter() - t2

        self._docs.extend(documents)

        t_total = time.perf_counter() - t0
        docs_per_sec = len(documents) / t_embed if t_embed > 0 else float("inf")
        print(f"[INDEX-TIMING] vector.index: {t_total:.3f}s (chunks={len(documents)}) | "
              f"prep={t_prep:.3f}s embed={t_embed:.3f}s [{docs_per_sec:.1f} docs/s] "
              f"upsert={t_upsert:.3f}s")

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Tuple[Document, float]]:
        query_embedding = self._embedding.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                score = 1.0 - distance
                docs.append((Document(page_content=text, metadata=meta), score))

        return docs

    def sync_from_docs(self, documents: list):
        """Bulk-load BM25 from existing documents (e.g. ChromaDB recovery)."""
        import time
        t0 = time.perf_counter()
        new_count = 0
        for doc in documents:
            cid = doc.metadata.get("chunk_id")
            if cid and cid in self._ids:
                continue
            if cid:
                self._ids.add(cid)
            tokens = self._tokenize(doc.page_content)
            doc_len = len(tokens)
            term_freq = {}
            for t in tokens:
                term_freq[t] = term_freq.get(t, 0) + 1
            self._docs.append(doc)
            self._doc_records.append((doc_len, term_freq))
            self._N += 1
            self._sum_doc_len += doc_len
            for term in term_freq:
                self._df[term] = self._df.get(term, 0) + 1
            new_count += 1
        t_total = time.perf_counter() - t0
        if new_count:
            print(f"[INDEX-TIMING] bm25.sync: {t_total:.3f}s (loaded={new_count}, total={self._N})")

    def clear_all(self):
        """Remove all documents from ChromaDB."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)

    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks with given source path from ChromaDB."""
        import time
        t0 = time.perf_counter()
        results = self._collection.get(
            where={"source": source_path},
            include=["metadatas"],
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        deleted = len(ids_to_delete)
        t_total = time.perf_counter() - t0
        print(f"[INDEX-TIMING] vector.delete: {t_total:.3f}s (deleted={deleted})")
        return deleted

    @property
    def collection_size(self) -> int:
        return self._collection.count()


# =====================================================================
#  Hybrid Retriever (BM25 + Vector)
# =====================================================================

class HybridRetriever:
    """
    Hybrid retriever that fuses BM25 keyword + vector semantic search
    using RRF (Reciprocal Rank Fusion).
    Vector search failure falls back to BM25-only gracefully.
    """

    def __init__(self):
        self.bm25 = BM25Engine()
        self.vector = VectorEngine()
        self._is_indexed = False
        # Recover BM25 from persistent ChromaDB if BM25 is empty
        self._sync_bm25_from_chromadb()

    def _sync_bm25_from_chromadb(self):
        if self.bm25.size > 0:
            return
        try:
            all_data = self.vector._collection.get(include=["documents","metadatas"])
            if all_data and all_data.get("ids"):
                from langchain_core.documents import Document
                seen = set()
                docs = []
                dup = []
                cids = all_data["ids"]
                for i in range(len(cids)):
                    meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
                    cid = meta.get("chunk_id", "")
                    if cid and cid in seen:
                        dup.append(cids[i])
                        continue
                    if cid:
                        seen.add(cid)
                    txt = all_data["documents"][i] if all_data["documents"] else ""
                    src = meta.get("source", "")
                    if src:
                        meta["source"] = src.replace(chr(92)*2, "/")
                    docs.append(Document(page_content=txt, metadata=meta))
                if dup:
                    self.vector._collection.delete(ids=dup)
                    print(f"[INIT] Removed {len(dup)} duplicate ChromaDB entries")
                if docs:
                    self.bm25.sync_from_docs(docs)
                    print(f"[INIT] Synced BM25 from ChromaDB: {self.bm25.size} docs")
        except Exception as e:
            print(f"[INIT] BM25 sync skipped ({e})")

    def index(self, documents: List[Document]):
        self.bm25.index(documents)
        try:
            self.vector.index(documents)
        except Exception as e:
            print(f"[WARN] Vector indexing failed ({e}), using BM25 only.")
        self._is_indexed = True
        print(f"[INFO] Indexed {len(documents)} new chunks. Total: BM25={self.bm25.size}, Vector={self.vector.collection_size}")

    @property
    def size(self) -> int:
        return self.bm25.size

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> List[Document]:
        recall_k = max(top_k * 3, 10)
        bm25_results = self.bm25.search(query, top_k=recall_k)
        vector_results = []

        try:
            vector_results = self.vector.search(query, top_k=recall_k)
        except Exception:
            pass

        if not vector_results:
            return [doc for doc, _ in bm25_results[:top_k]]

        return self._rrf_fusion(bm25_results, vector_results, top_k, bm25_weight, vector_weight)

    @staticmethod
    def _rrf_fusion(
        bm25_results: List[Tuple[Document, float]],
        vector_results: List[Tuple[Document, float]],
        top_k: int,
        bm25_weight: float,
        vector_weight: float,
    ) -> List[Document]:
        k = 60
        scores = {}

        for rank, (doc, score) in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf = bm25_weight / (k + rank + 1)
            if chunk_id not in scores:
                scores[chunk_id] = {"doc": doc, "score": 0.0}
            scores[chunk_id]["score"] += rrf

        for rank, (doc, score) in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf = vector_weight / (k + rank + 1)
            if chunk_id not in scores:
                scores[chunk_id] = {"doc": doc, "score": 0.0}
            scores[chunk_id]["score"] += rrf

        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items[:top_k]]

    def sync_from_docs(self, documents: list):
        """Bulk-load BM25 from existing documents (e.g. ChromaDB recovery)."""
        import time
        t0 = time.perf_counter()
        new_count = 0
        for doc in documents:
            cid = doc.metadata.get("chunk_id")
            if cid and cid in self._ids:
                continue
            if cid:
                self._ids.add(cid)
            tokens = self._tokenize(doc.page_content)
            doc_len = len(tokens)
            term_freq = {}
            for t in tokens:
                term_freq[t] = term_freq.get(t, 0) + 1
            self._docs.append(doc)
            self._doc_records.append((doc_len, term_freq))
            self._N += 1
            self._sum_doc_len += doc_len
            for term in term_freq:
                self._df[term] = self._df.get(term, 0) + 1
            new_count += 1
        t_total = time.perf_counter() - t0
        if new_count:
            print(f"[INDEX-TIMING] bm25.sync: {t_total:.3f}s (loaded={new_count}, total={self._N})")

    def clear_all(self):
        """Remove all documents from both engines."""
        self.bm25.clear_all()
        try:
            self.vector.clear_all()
        except Exception as e:
            print(f"[WARN] Vector clear failed ({e})")
        print("[INFO] Knowledge base cleared")

    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a given source file from both engines."""
        n1 = self.bm25.delete_by_source(source_path)
        n2 = 0
        try:
            n2 = self.vector.delete_by_source(source_path)
        except Exception as e:
            print(f"[WARN] Vector delete failed ({e})")
        total = max(n1, n2)
        print(f"[INFO] Deleted from {source_path}: BM25={n1}, Vector={n2}")
        return total

    def clear_all(self):
        """Remove all documents from both engines."""
        self.bm25.clear_all()
        try:
            self.vector.clear_all()
        except Exception as e:
            print(f"[WARN] Vector clear failed ({e})")
        print("[INFO] Knowledge base cleared")

    def get_indexed_sources(self) -> list:
        """Return list of unique indexed source paths."""
        return self.bm25.get_indexed_sources()

    def get_context(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> Tuple[str, List[Document]]:
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return "", []

        context_parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, docs


# ========== Global Singleton ==========

_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
