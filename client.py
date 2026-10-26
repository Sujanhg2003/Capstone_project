import requests

url = "http://localhost:11434/api/embed"

prompt = "Machine learning is a branch of artificial intelligence"

payload = {
    "model":"nomic-embed-text",
    "input":prompt
}

response = requests.post(
    url,
    json=payload
)

response.raise_for_status()

data = response.json()

print("LLM Response:")
print(data["embeddings"])