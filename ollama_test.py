import requests

OLLAMA_URL = "http://172.16.17.127:11434"

response = requests.post(
    f"{OLLAMA_URL}/api/generate",
    json={
        "model": "qwen3.5:0.8b",
        "prompt": "Why is the sky blue?",
        "stream": False
    }
)

result = response.json()
# print(response.status_code)
print(result["response"])