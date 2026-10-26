import os

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

PRIMARY_MODEL = os.getenv(
    "PRIMARY_MODEL",
    "qwen3:8b"
    
)

FALLBACK_MODEL = os.getenv(
    "FALLBACK_MODEL",
    "llama3.2:1b"
)



REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 1

# Orchestration limits
MAX_TOOL_STEPS = 5
MAX_TOOL_CALLS = 10

# Input and generation limits
MAX_INPUT_CHARS = 4000
MAX_OUTPUT_TOKENS = 512
NUM_CTX = 4096

MAX_FINAL_ANSWER_CHARS = 4000