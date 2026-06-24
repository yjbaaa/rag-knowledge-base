"""
HyDE (Hypothetical Document Embeddings)
Generates a hypothetical answer to bridge the vocabulary gap
between short queries and document content.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_CHAT_MODEL


HYDE_SYSTEM_PROMPT = """You are a helpful assistant. Given a question, write a short passage that answers the question.

Rules:
- Write as if you are a knowledgeable expert answering the question.
- Keep the answer concise (2-4 sentences).
- Use factual, information-dense language similar to what would appear in a technical document.
- Do NOT preface with "The answer is..." or similar - just output the passage directly.
- If you do not know the answer, make a reasonable, educated guess that sounds like a document excerpt."""


class HyDE:
    """HyDE: generates hypothetical document for query enhancement."""

    def __init__(self):
        self._llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            temperature=0.3,
            timeout=15,
            max_retries=1,
        )

    def generate(self, query: str) -> str:
        """
        Generate a hypothetical document passage based on the query.

        Returns the generated passage, or the original query on failure.
        """
        messages = [
            SystemMessage(content=HYDE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        try:
            response = self._llm.invoke(messages)
            passage = response.content.strip()
            return passage if passage else query
        except Exception:
            return query


# ========== Global singleton ==========

_hyde_instance: Optional[HyDE] = None


def get_hyde() -> HyDE:
    global _hyde_instance
    if _hyde_instance is None:
        _hyde_instance = HyDE()
    return _hyde_instance


def generate_hypothetical_doc(query: str) -> str:
    """Convenience: generate HyDE passage."""
    return get_hyde().generate(query)
