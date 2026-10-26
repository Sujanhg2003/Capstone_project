import requests
import json


url = "http://172.16.17.127:11434/api/generate"

payload = {
    "model": "qwen3.5:0.8b",
    "prompt": "Explain RAG in one sentence.",
    "stream": True
}

response = requests.post(url, json=payload,stream=True)

response.raise_for_status()

for line in response.iter_lines(decode_unicode=True):
    if line:
        data = json.loads(line)
        print(data.get("response", ""), end="", flush=True)