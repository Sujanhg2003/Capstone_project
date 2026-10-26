import requests

OLLAMA_URL = "http://172.16.17.127:11434"

user_prompt = "Explain neural networks."

system_prompts = [
    "You are a teacher for beginners. Use very simple language.",
    
    "You are a deep learning researcher. Use technical terminology.",
    
    "You are a Python developer. Include a Python example.",
    
]


for i , system_prompt in enumerate(system_prompts,start=1):
    print(f"EXPERIMENT {i}")
    print("=" * 60)

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json= {
            "model": "qwen3.5:0.8b",
            "messages":[
                {
                    "role":"system",
                    "content": system_prompt
                },
                {
                "role":"user",
                "content": user_prompt
                }
            ],
            "stream":False
        }
    )

    response.raise_for_status()

    result = response.json()

    print("SYSTEM:")
    print(system_prompt)

    print("\nUSER:")
    print(user_prompt)

    print("\nASSISTANT:")
    print(result["message"]["content"])