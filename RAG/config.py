from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "data"

CHROMA_DIR = BASE_DIR / "chroma_db"

OLLAMA_BASE_URL = "http://localhost:11434"

LLM_MODEL = "qwen3:8b"

EMBEDDING_MODEL = "nomic-embed-text"

COLLECTION_NAME = "fir_documents"

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150

TOP_K = 5

