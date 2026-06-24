"""
Enterprise RAG Knowledge Base - Global Configuration
Set via .env file or environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ========== Project Root ==========
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Auto-load .env from project root.
# override=True: .env is the authoritative config source, so it takes
# precedence over any pre-existing process-level env var (e.g. a stale
# OPENAI_API_KEY left in the shell). Set override=False if you'd rather
# let the shell environment win.
_load_dotenv = load_dotenv(ROOT_DIR / ".env", override=True)

# ========== Path Configuration ==========
DATA_DIR = ROOT_DIR / "data"
DB_DIR = ROOT_DIR / "db"
LOG_DIR = ROOT_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========== LLM / Embedding Configuration ==========
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-key-here")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ========== Vector DB Configuration ==========
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_knowledge_base")

# ========== Chunking Configuration ==========
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_SEPARATORS = ["\n\n", "\n", "\u3002", ".", "\uff01", "\uff1f", " ", ""]

# ========== Retrieval Configuration ==========
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"

# ========== Memory Configuration ==========
MAX_MEMORY_TOKENS = int(os.getenv("MAX_MEMORY_TOKENS", "4000"))
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))

# ========== API Configuration ==========
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ========== Streamlit Configuration ==========
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
