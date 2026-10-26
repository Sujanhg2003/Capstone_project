import os

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SEMANTIC_TOP_K = 20
KEYWORD_TOP_K = 20
RERANK_TOP_K = 3
RRF_K = 60

TOP_K=3
RETRIEVAL_DISTANCE_THRESHOLD=1.0

TEMPERATURE = 0.0

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:8b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text:latest"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./data/chroma"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "company_documents"
)


COHERE_API_KEY = os.getenv(
    "COHERE_API_KEY"
)

COHERE_RERANK_MODEL = os.getenv(
    "COHERE_RERANK_MODEL",
    "rerank-v4.0-fast"
)