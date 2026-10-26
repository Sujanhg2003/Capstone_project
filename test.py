import requests
import json

URL = "http://172.16.17.127:11434/api/generate"

payload = {
    "model": "qwen3.5:0.8b",
    "prompt": "Explain RAG in one sentence.",
    "stream": False,
    "think": False,
    "options": {
        "temperature": 1.0,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": 150,
        "seed": 100
    }
}

response = requests.post(URL, json=payload)

print("STATUS:", response.status_code)

print("\nTOKENS:")
print("Prompt tokens :", response.json().get("prompt_eval_count"))
print("Output tokens :", response.json().get("eval_count"))

print("RAW RESPONSE:")
print(response.json()["response"])

