import json
import time
from pathlib import Path


from rag import ask_fir_question
from evaluation.ollama_judge import OllamaJudge


DATASET_PATH = (
    Path(__file__).parent /
    "fir_evaluation_dataset.json"
)

judge_model = OllamaJudge(
    model = "qwen3:8b"
)

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

def run_rag(question):

    start = time.perf_counter()

    result = ask_fir_question(question)

    latency = time.perf_counter() - start

    return {
        "answer": result["answer"],
        "context": result["context"],
        "latency": latency,
    }


def main():

    dataset = load_dataset()[:1]

    test_cases = []

    metadata = []

    print("\n======================================")
    print("       DEEPEVAL RAG EVALUATION")
    print("======================================\n")

    for i, item in enumerate(dataset[:1], start=1):

        question = item["question"]

        print(
            f"[{i}/{len(dataset)}] "
            f"{question}"
        )

        result = run_rag(question)

        test_case = LLMTestCase(
            input=question,
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=result["context"],
        )

        test_cases.append(test_case)

        metadata.append({
            "id": item["id"],
            "question": question,
            "category": item["category"],
            "difficulty": item["difficulty"],
            "latency": result["latency"],
            "actual_answer": result["answer"],
            "expected_answer": item["expected_answer"],
        })

    # ========================================================
    # METRICS
    # ========================================================

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=judge_model,
        include_reason=True,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge_model,
        include_reason=True,
    )

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        model=judge_model,
        include_reason=True,
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    print("\nRunning DeepEval...\n")

    evaluate(
        test_cases=test_cases,
        metrics=[
            faithfulness,
            answer_relevancy,
            contextual_relevancy,
        ],
    )

    # ========================================================
    # SAVE BASIC RESULTS
    # ========================================================

    output_path = (
        Path(__file__).parent /
        "results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()