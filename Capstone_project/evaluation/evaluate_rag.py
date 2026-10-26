import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import LLM_MODEL


# ============================================================
# CONFIGURATION
# ============================================================

JSON_FILE = r"C:\Learning\Capstone_project\evaluation\results_with_context.json"

OLLAMA_URL = "http://localhost:11434"

# LLM used only for faithfulness evaluation
LLM_MODEL = LLM_MODEL

# Embedding model for answer relevancy
EMBED_MODEL = "nomic-embed-text"

# Number of questions to evaluate
NUM_QUESTIONS = 10


# ============================================================
# OLLAMA LLM
# ============================================================

def ollama_generate(prompt):

    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False
    }

    response = requests.post(
        url,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]


# ============================================================
# OLLAMA EMBEDDING
# ============================================================

def get_embedding(text):

    url = f"{OLLAMA_URL}/api/embed"

    payload = {
        "model": EMBED_MODEL,
        "input": text
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return np.array(data["embeddings"][0])


# ============================================================
# 1. FAITHFULNESS
# ============================================================

def calculate_faithfulness(question, answer, contexts):

    context_text = "\n\n".join(
        f"Context {i+1}:\n{ctx}"
        for i, ctx in enumerate(contexts)
    )

    prompt = f"""
You are evaluating a RAG system.

Your task is to measure FAITHFULNESS.

Faithfulness means:
The answer must be completely supported by the provided context.

Do NOT judge whether the answer is generally correct.
Only check whether the claims in the answer are supported by the context.

Question:
{question}

Retrieved Context:
{context_text}

Answer:
{answer}

Give a score between 0 and 1.

0 = completely unsupported
0.25 = mostly unsupported
0.5 = partially supported
0.75 = mostly supported
1 = completely supported

Return ONLY a number between 0 and 1.
"""

    result = ollama_generate(prompt).strip()

    try:
        score = float(result)
        return max(0.0, min(1.0, score))
    except ValueError:
        print("Invalid faithfulness response:", result)
        return 0.0


# ============================================================
# 2. CONTEXT PRECISION
# ============================================================

def calculate_context_precision(item):

    retrieved_chunks = item.get("retrieved_chunks", [])
    expected_sources = item.get("expected_sources", [])

    if not retrieved_chunks:
        return 0.0

    expected = {
        (
            source["document"],
            source["page"]
        )
        for source in expected_sources
    }

    relevant = 0
    precision_sum = 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):

        chunk_source = (
            chunk.get("document"),
            chunk.get("page")
        )

        if chunk_source in expected:

            relevant += 1

            precision_at_k = relevant / rank

            precision_sum += precision_at_k

    if relevant == 0:
        return 0.0

    return precision_sum / relevant


# ============================================================
# 3. ANSWER RELEVANCY
# ============================================================

def calculate_answer_relevancy(question, answer):

    question_embedding = get_embedding(question)

    answer_embedding = get_embedding(answer)

    similarity = cosine_similarity(
        [question_embedding],
        [answer_embedding]
    )[0][0]

    # Cosine similarity is normally [-1,1]
    # Convert to 0-1 range

    score = (similarity + 1) / 2

    return float(score)


# ============================================================
# LOAD DATASET
# ============================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:

    data = json.load(f)


print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)

print(f"Total questions in dataset: {len(data)}")


# ============================================================
# SELECT QUESTIONS
# ============================================================

# Use first 10 questions
# You can change this to random sampling if required.

evaluation_data = data["results"][:NUM_QUESTIONS]

print(f"Questions evaluated: {len(evaluation_data)}")
print()


# ============================================================
# EVALUATION
# ============================================================

results = []


for index, item in enumerate(evaluation_data, start=1):

    question = item["question"]
    answer = item["answer"]

    contexts = [
        chunk["text"]
        for chunk in item.get("retrieved_chunks", [])
    ]

    print("-" * 70)
    print(f"Question {index}/{len(evaluation_data)}")
    print(question)

    # --------------------------------------------------------
    # Faithfulness
    # --------------------------------------------------------

    faithfulness = calculate_faithfulness(
        question,
        answer,
        contexts
    )

    print(f"Faithfulness      : {faithfulness:.3f}")

    # --------------------------------------------------------
    # Context Precision
    # --------------------------------------------------------

    context_precision = calculate_context_precision(item)

    print(f"Context Precision : {context_precision:.3f}")

    # --------------------------------------------------------
    # Answer Relevancy
    # --------------------------------------------------------

    answer_relevancy = calculate_answer_relevancy(
        question,
        answer
    )

    print(f"Answer Relevancy  : {answer_relevancy:.3f}")

    results.append({
        "id": item.get("id"),
        "question": question,
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "answer_relevancy": answer_relevancy
    })


# ============================================================
# FINAL RESULTS
# ============================================================

avg_faithfulness = np.mean([
    r["faithfulness"]
    for r in results
])

avg_context_precision = np.mean([
    r["context_precision"]
    for r in results
])

avg_answer_relevancy = np.mean([
    r["answer_relevancy"]
    for r in results
])


print()
print("=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

print(f"Questions evaluated : {len(results)}")

print(
    f"Faithfulness        : {avg_faithfulness:.3f}"
)

print(
    f"Context Precision   : {avg_context_precision:.3f}"
)

print(
    f"Answer Relevancy    : {avg_answer_relevancy:.3f}"
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = "rag_evaluation_results.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "num_questions": len(results),
            "average_metrics": {
                "faithfulness": avg_faithfulness,
                "context_precision": avg_context_precision,
                "answer_relevancy": avg_answer_relevancy
            },
            "questions": results
        },
        f,
        indent=4
    )


print()
print(f"Results saved to: {output_file}")