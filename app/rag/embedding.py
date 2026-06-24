"""
Embedding 模型封装
支持 OpenAI / 本地 BGE 两种模式，策略模式切换
"""

from typing import List
from langchain_openai import OpenAIEmbeddings
from app.core.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_API_BASE,
    OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM,
    LOCAL_EMBEDDING_MODEL,
)


class BaseEmbedding:
    """Embedding 抽象基类"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class OpenAIEmbeddingWrapper(BaseEmbedding):
    """OpenAI Embedding 封装"""

    def __init__(self):
        self._model = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            dimensions=OPENAI_EMBEDDING_DIM,
            timeout=5,        # 5s timeout to avoid hanging
            max_retries=0,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    @property
    def dimension(self) -> int:
        return OPENAI_EMBEDDING_DIM


class HuggingFaceEmbeddingWrapper(BaseEmbedding):
    """本地 HuggingFace Embedding 封装 (BGE 系列)"""

    def __init__(self):
        import os
        # Use HF mirror for China access
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from langchain_huggingface import HuggingFaceEmbeddings
        self._model = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    @property
    def dimension(self) -> int:
        # BGE-small-zh-v1.5: 512; BGE-base-zh-v1.5: 768; BGE-large-zh-v1.5: 1024
        return 512


# ========== 工厂函数 ==========

_embedding_instance: BaseEmbedding = None


def get_embedding() -> BaseEmbedding:
    """获取 Embedding 实例（单例）"""
    global _embedding_instance
    if _embedding_instance is None:
        if LLM_PROVIDER == "openai":
            _embedding_instance = OpenAIEmbeddingWrapper()
        else:
            _embedding_instance = HuggingFaceEmbeddingWrapper()
    return _embedding_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """便捷函数：批量向量化"""
    return get_embedding().embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """便捷函数：单条查询向量化"""
    return get_embedding().embed_query(text)
