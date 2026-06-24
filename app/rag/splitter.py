"""
智能文本分块器
基于递归字符分割，针对中文场景定制分隔符
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


class DocSplitter:
    """
    文档分块器，封装 LangChain RecursiveCharacterTextSplitter
    - 中文友好的分隔符顺序
    - 保留原始元数据，新增 chunk_index / chunk_id
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or CHUNK_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=False,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        all_chunks = []
        for doc in documents:
            chunks = self._splitter.split_text(doc.page_content)
            for i, chunk_text in enumerate(chunks):
                meta = dict(doc.metadata)
                meta.update({
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "chunk_id": self._make_chunk_id(doc.metadata.get("source", "unknown"), i),
                })
                all_chunks.append(Document(page_content=chunk_text, metadata=meta))
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata["global_index"] = idx
            # Regenerate chunk_id using global_index for guaranteed uniqueness
            chunk.metadata["chunk_id"] = self._make_chunk_id(
                chunk.metadata.get("source", "unknown"), idx
            )
        return all_chunks

    def split_single(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        doc = Document(page_content=text, metadata=metadata or {})
        return self.split([doc])

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int) -> str:
        import hashlib
        raw = f"{source}#chunk{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """One-liner to split documents."""
    return DocSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(documents)
