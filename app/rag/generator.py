# -*- coding: utf-8 -*-
"""Answer Generator - supports per-request API configuration"""

from typing import List, Optional, Generator, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.core.config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_CHAT_MODEL
from app.memory.memory import ConversationMemory
from app.utils.citation import CitationTracer, format_citation_card

RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant for an enterprise knowledge base. Answer the user's question based on the provided context. You MUST answer in Simplified Chinese (简体中文).

Rules:
1. Answer ONLY using information from the context below. If the context doesn't contain the answer, say "提供的文档中未包含相关信息。"
2. Cite sources inline using [1], [2] format matching the source numbers in the context.
3. Be concise and accurate. Use bullet points for lists when appropriate.
4. If the question is ambiguous, ask for clarification rather than guessing.
5. Always respond in Chinese (简体中文), regardless of the language of the source documents."""

class AnswerGenerator:
    def __init__(self, api_key=None, api_base=None, model=None):
        self.api_key = api_key or OPENAI_API_KEY
        self.api_base = api_base or OPENAI_API_BASE
        self.model = model or OPENAI_CHAT_MODEL

    @property
    def _llm(self):
        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=0.3,
            timeout=60,
            max_retries=1,
        )

    def generate(self, context, query, chat_history=None):
        messages = self._build_messages(context, query, chat_history)
        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"[Error generating answer: {e}]"

    def generate_stream(self, context, query, chat_history=None):
        messages = self._build_messages(context, query, chat_history)
        try:
            for chunk in self._llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[Error: {e}]"

    def generate_with_sources(self, context, query, sources, chat_history=None, api_key=None, api_base=None, model=None):
        key = api_key or self.api_key
        base = api_base or self.api_base
        mdl = model or self.model
        llm = ChatOpenAI(model=mdl, api_key=key, base_url=base, temperature=0.3, timeout=60, max_retries=1)
        messages = self._build_messages(context, query, chat_history)
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"[Error generating answer: {e}]"

    def generate_stream_with_sources(self, context, query, sources, chat_history=None, api_key=None, api_base=None, model=None):
        key = api_key or self.api_key
        base = api_base or self.api_base
        mdl = model or self.model
        llm = ChatOpenAI(model=mdl, api_key=key, base_url=base, temperature=0.3, timeout=60, max_retries=1)
        messages = self._build_messages(context, query, chat_history)
        try:
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[Error: {e}]"

    def _build_messages(self, context, query, chat_history=None):
        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)]
        if chat_history and not chat_history.is_empty:
            history_messages = chat_history.get_langchain_messages(last_n=6)
            messages.extend(history_messages)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        messages.append(HumanMessage(content=user_prompt))
        return messages

    @staticmethod
    def trace_citations(answer, sources):
        tracer = CitationTracer()
        return tracer.parse(answer, sources)

def get_generator(api_key=None, api_base=None, model=None):
    return AnswerGenerator(api_key=api_key, api_base=api_base, model=model)
