"""
RAG Pipeline - Main orchestrator
Wires together: Load -> Split -> Index -> Rewrite -> HyDE -> Retrieve -> Rerank -> Generate
"""

import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.rag.loader import DocumentLoader, load_documents
from app.rag.splitter import DocSplitter
from app.rag.retriever import get_retriever, HybridRetriever
from app.rag.rewriter import get_rewriter, QueryRewriter
from app.rag.hyde import get_hyde, HyDE
from app.rag.reranker import get_reranker, Reranker
from app.rag.generator import get_generator, AnswerGenerator
from app.utils.citation import CitationTracer, CitationResult, trace_citations
from app.memory.memory import ConversationMemory, get_session_manager, SessionManager
from app.core.config import (
    RETRIEVAL_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ENABLE_RERANKER,
)


@dataclass
class RAGResult:
    """Structured RAG query result."""
    query: str
    rewritten_query: str = ""
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    context_chunks: List[Document] = field(default_factory=list)
    session_id: str = ""
    citation_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "sources": self.sources,
            "source_count": len(self.sources),
            "session_id": self.session_id,
        }
        if self.citation_result:
            d["citation_result"] = self.citation_result
        return d


class RAGPipeline:
    """
    Full RAG pipeline orchestrator.

    Usage:
        pipeline = RAGPipeline()
        pipeline.index_files(["doc1.pdf", "doc2.md"])
        result = pipeline.query("What is RAG?")
    """

    def __init__(
        self,
        enable_rewrite: bool = True,
        enable_hyde: bool = False,
        enable_rerank: bool = ENABLE_RERANKER,
        top_k: int = RETRIEVAL_TOP_K,
    ):
        self.enable_rewrite = enable_rewrite
        self.enable_hyde = enable_hyde
        self.enable_rerank = enable_rerank
        self.top_k = top_k

        # Lazy-init components
        self._retriever: Optional[HybridRetriever] = None
        self._rewriter: Optional[QueryRewriter] = None
        self._hyde: Optional[HyDE] = None
        self._reranker: Optional[Reranker] = None
        self._generator: Optional[AnswerGenerator] = None
        self._session_mgr: Optional[SessionManager] = None

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    @property
    def rewriter(self) -> QueryRewriter:
        if self._rewriter is None:
            self._rewriter = get_rewriter()
        return self._rewriter

    @property
    def hyde(self) -> HyDE:
        if self._hyde is None:
            self._hyde = get_hyde()
        return self._hyde

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = get_generator()
        return self._generator

    @property
    def session_mgr(self) -> SessionManager:
        if self._session_mgr is None:
            self._session_mgr = get_session_manager()
        return self._session_mgr

    # =================================================================
    #  Document Indexing
    # =================================================================

    def index_files(
        self,
        file_paths: List[str],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> int:
        """
        Load, split, and index multiple files into the hybrid retriever.

        Returns: number of chunks indexed.
        """
        # Load
        t0 = time.perf_counter()
        docs = load_documents(file_paths)
        t_load = time.perf_counter() - t0
        print(f"[INDEX-TIMING] load: {t_load:.3f}s (files={len(file_paths)}, docs={len(docs)})")
        if not docs:
            print("[WARN] No documents loaded.")
            return 0

        # Split
        t1 = time.perf_counter()
        splitter = DocSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split(docs)
        t_split = time.perf_counter() - t1
        print(f"[INDEX-TIMING] split: {t_split:.3f}s (docs={len(docs)} -> chunks={len(chunks)})")

        # Index
        t2 = time.perf_counter()
        self.retriever.index(chunks)
        t_index = time.perf_counter() - t2
        t_total = time.perf_counter() - t0
        print(f"[INDEX-TIMING] index: {t_index:.3f}s (chunks={len(chunks)})")
        print(f"[INDEX-TIMING] TOTAL: {t_total:.3f}s "
              f"(load={t_load:.3f}s, split={t_split:.3f}s, index={t_index:.3f}s)")
        print(f"[INFO] Indexed {len(chunks)} chunks from {len(file_paths)} file(s).")
        return len(chunks)

    def index_documents(self, documents: List[Document]) -> int:
        """Index pre-loaded documents directly."""
        splitter = DocSplitter()
        chunks = splitter.split(documents)
        self.retriever.index(chunks)
        return len(chunks)

    # =================================================================
    #  Query Pipeline
    # =================================================================

    def query(
        self,
        query: str,
        chat_history: str = "",
        top_k: Optional[int] = None,
    ) -> RAGResult:
        """
        Execute the full RAG query pipeline.

        Flow: Rewrite -> [HyDE] -> Retrieve -> [Rerank] -> Build Context

        Args:
            query: The user's question.
            chat_history: Optional conversation history for context-aware rewrite.
            top_k: Override number of results.

        Returns:
            RAGResult with context and source documents (answer filled by generator).
        """
        top_k = top_k or self.top_k
        result = RAGResult(query=query)

        # Step 1: Query Rewriting
        working_query = query
        if self.enable_rewrite:
            working_query = self.rewriter.rewrite_with_context(query, chat_history)
            result.rewritten_query = working_query

        # Step 2: HyDE (optional)
        search_query = working_query
        if self.enable_hyde:
            hyde_passage = self.hyde.generate(working_query)
            if hyde_passage and hyde_passage != working_query:
                search_query = hyde_passage

        # Step 3: Hybrid Retrieval
        docs = self.retriever.retrieve(search_query, top_k=top_k * 2 if self.enable_rerank else top_k)

        # Step 4: Reranking (optional)
        if self.enable_rerank and docs:
            docs = self.reranker.rerank(working_query, docs, top_k=top_k)

        # Step 5: Build result
        result.context_chunks = docs[:top_k]
        result.sources = [self._doc_to_source(doc) for doc in result.context_chunks]

        return result

    def get_context(self, query: str, chat_history: str = "", top_k: Optional[int] = None) -> str:
        """
        Convenience: run query pipeline and return formatted context string.
        """
        result = self.query(query, chat_history, top_k)
        if not result.context_chunks:
            return ""

        parts = []
        for i, doc in enumerate(result.context_chunks, start=1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    # =================================================================
    #  Query + Memory + Generate (full pipeline with answer)
    # =================================================================

    def ask(
        self,
        query: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
        api_key: str = "",
        api_base: str = "",
        model: str = "",
    ) -> RAGResult:
        """
        Full RAG pipeline with multi-turn memory and answer generation.

        1. Retrieve relevant context
        2. Generate answer with LLM
        3. Update conversation memory
        4. Return result with sources + answer

        Args:
            query: User question.
            session_id: Session identifier for multi-turn conversation.
            top_k: Number of documents to retrieve.

        Returns:
            RAGResult with answer, sources, and context.
        """
        # Get session memory
        memory = self.session_mgr.get_session(session_id)
        chat_history = memory.get_formatted_history()

        # 1. Retrieval (with context-aware rewrite if enabled).
        #    query() already fills result.context_chunks / result.sources.
        result = self.query(query, chat_history=chat_history, top_k=top_k)

        # 2. Build context string from already-retrieved chunks (no re-retrieval)
        context = self._format_context(result.context_chunks)

        # 3. Generate answer (must happen BEFORE citation parsing)
        if context:
            result.answer = self.generator.generate_with_sources(
                context=context,
                query=result.rewritten_query or query,
                sources=result.sources,
                chat_history=memory,
            )

            # 4. Parse inline citation markers from the GENERATED answer
            tracer = CitationTracer()
            citation = tracer.parse(result.answer, result.sources)
            result.citation_result = {
                "citation_count": citation.citation_count,
                "cited_markers": citation.cited_markers,
                "uncited_sources": citation.uncited_sources,
                "has_citations": citation.has_citations,
                "citation_markdown": tracer.format_markdown(citation) if citation.has_citations else "",
            }
        else:
            result.answer = "No relevant documents found in the knowledge base."

        # 5. Update memory
        memory.add_turn(query, result.answer)
        self.session_mgr.save_session(session_id)

        result.session_id = session_id
        return result

    def ask_stream(
        self,
        query: str,
        session_id: str = "default",
        top_k: Optional[int] = None,
        api_key: str = "",
        api_base: str = "",
        model: str = "",
    ):
        """
        Streaming version of ask(). Yields a meta event first, then text chunks.

        Event types:
            - dict  -> metadata event ({"sources": [...], "rewritten_query": ...})
            - str   -> answer text chunk

        Usage:
            for chunk in pipeline.ask_stream("What is RAG?"):
                if isinstance(chunk, dict):
                    sources = chunk["sources"]
                else:
                    print(chunk, end="")  # token
        """
        memory = self.session_mgr.get_session(session_id)
        chat_history = memory.get_formatted_history()

        # Retrieval (single pass). Reuse the chunks already fetched by query().
        result = self.query(query, chat_history=chat_history, top_k=top_k)
        context = self._format_context(result.context_chunks)

        # Emit metadata (sources) UP FRONT so the client needs only ONE request.
        if result.sources:
            yield {
                "sources": result.sources,
                "rewritten_query": result.rewritten_query,
                "source_count": len(result.sources),
            }

        if not context:
            answer = "No relevant documents found."
            memory.add_turn(query, answer)
            self.session_mgr.save_session(session_id)
            yield answer
            return

        # Stream answer
        full_answer = []
        for chunk in self.generator.generate_stream_with_sources(
            context=context,
            query=result.rewritten_query or query,
            sources=result.sources,
            chat_history=memory,
            api_key=api_key,
            api_base=api_base,
            model=model,
        ):
            full_answer.append(chunk)
            yield chunk

        # Update memory after streaming completes
        answer_text = "".join(full_answer)
        memory.add_turn(query, answer_text)
        self.session_mgr.save_session(session_id)

    def clear_all(self):
        """Remove all documents from the knowledge base."""
        self.retriever.clear_all()

    def delete_files(self, file_paths: list) -> int:
        """Delete all chunks from given files. Returns total chunks deleted."""
        total = 0
        for fp in file_paths:
            total += self.retriever.delete_by_source(str(fp))
        return total

    def get_indexed_sources(self) -> list:
        """Return list of unique indexed source paths."""
        return self.retriever.get_indexed_sources()

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self.session_mgr.delete_session(session_id)

    # =================================================================
    #  Helpers
    # =================================================================

    @staticmethod
    def _doc_to_source(doc: Document) -> Dict[str, Any]:
        """Extract source citation info from a document."""
        meta = doc.metadata
        return {
            "content": doc.page_content[:200],
            "source": meta.get("source", "unknown"),
            "filename": meta.get("filename", "unknown"),
            "file_type": meta.get("file_type", ""),
            "page": meta.get("page"),
            "chunk_id": meta.get("chunk_id", ""),
            "chunk_index": meta.get("chunk_index"),
        }

    @staticmethod
    def _format_context(chunks: List[Document]) -> str:
        """Format already-retrieved chunks into a context string for the LLM.

        Unlike get_context(), this does NOT re-run retrieval — it only formats
        the chunks produced by query(). Use this inside ask()/ask_stream().
        """
        if not chunks:
            return ""
        parts = []
        for i, doc in enumerate(chunks, start=1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


# ========== Global singleton ==========

_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
