import requests

OLLAMA_URL = "http://172.16.17.127:11434/api/generate"
MODEL = "qwen3.5:0.8b"


PROMPT = """
Explain what RAG is in simple terms.
Give one example and mention two advantages.
"""

def run(options):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "think": False,
            "options": options
        }
    )

    data = response.json()

    print("\n" + "=" * 60)
    print("PARAMETERS:", options)
    print("=" * 60)


    # print("=========================")
    # print(data)

    print("\nOUTPUT:")
    print(data["response"])

    print("\nTOKENS:")
    print("Prompt tokens :", data.get("prompt_eval_count"))
    print("Output tokens :", data.get("eval_count"))

base = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_predict": 150,
    "seed": 42
}

print("\n\n### TEMPERATURE ###")

for value in [0.2, 0.7, 1.2]:

    options = base.copy()
    options["temperature"] = value

    run(options)

print("\n\n### TOP-K ###")

for value in [1, 10, 40]:

    options = base.copy()
    options["top_k"] = value

    run(options)

print("\n\n### TOP-P ###")

for value in [0.3, 0.7, 0.9]:

    options = base.copy()
    options["top_p"] = value

    run(options)


print("\n\n### REPETITION PENALTY ###")

for value in [1.0, 1.1, 1.5]:

    options = base.copy()
    options["repeat_penalty"] = value

    run(options)


print("\n\n### MAX TOKENS ###")

for value in [30, 75, 150]:

    options = base.copy()
    options["num_predict"] = value

    run(options)

print("\n\n### SEED ###")

for value in [1, 42, 100]:

    options = base.copy()
    options["seed"] = value

    run(options)