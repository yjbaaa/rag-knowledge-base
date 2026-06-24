"""
Query Rewriter - LLM-based query optimization
Transforms raw user queries into clearer, more searchable forms.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_CHAT_MODEL


REWRITE_SYSTEM_PROMPT = """You are a query optimization expert. Your task is to rewrite the user's question into a clearer, more specific, self-contained version that is optimized for document retrieval. Keep the query in Chinese (简体中文).

Rules:
1. Expand ambiguous pronouns and implicit references into explicit terms.
2. Keep the query in Chinese (简体中文). If the query contains English technical terms, keep those as-is.
3. Keep technical terms exactly as-is (product names, codes, abbreviations).
4. Output ONLY the rewritten query, nothing else - no explanations, no prefixes.
5. If the query is already clear and well-formed, return it unchanged."""


class QueryRewriter:
    """LLM-based query rewriter for RAG retrieval optimization."""

    def __init__(self):
        self._llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            temperature=0.0,
            timeout=10,
            max_retries=1,
        )

    def rewrite(self, query: str) -> str:
        """Rewrite a single query."""
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        try:
            response = self._llm.invoke(messages)
            rewritten = response.content.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception:
            return query  # Fallback to original query on error

    def rewrite_with_context(self, query: str, chat_history: str = "") -> str:
        """Rewrite query considering conversation context for multi-turn dialogue."""
        if not chat_history.strip():
            return self.rewrite(query)

        prompt = f"""Given the conversation history below, rewrite the user's latest question into a self-contained query suitable for document retrieval. Resolve any pronouns and implicit references.

Conversation History:
{chat_history}

Latest Question: {query}

Rewritten Query:"""

        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        try:
            response = self._llm.invoke(messages)
            rewritten = response.content.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception:
            return query


# ========== Global singleton ==========

_rewriter_instance: Optional[QueryRewriter] = None


def get_rewriter() -> QueryRewriter:
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter()
    return _rewriter_instance


def rewrite_query(query: str, chat_history: str = "") -> str:
    """Convenience: rewrite a query."""
    return get_rewriter().rewrite_with_context(query, chat_history)
