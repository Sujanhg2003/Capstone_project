import json
import time
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

from rag import ask_fir_question

DATASET_PATH = (
    Path(__file__).parent /
    "fir_evaluation_dataset.json"
)

RESULT_PATH = (
    Path(__file__).parent /
    "ragas_results.json"
)


judge_llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
)

evaluator_llm = LangchainLLMWrapper(
    judge_llm
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
        "scores": result.get("scores", []),
        "latency": latency,
    }


def main():

    print("\n======================================")
    print("        RAGAS RAG EVALUATION")
    print("======================================\n")

    source_dataset = load_dataset()

    # --------------------------------------------
    # TEST ONLY ONE CASE FIRST
    # --------------------------------------------

    source_dataset = source_dataset[:1]

    samples = []

    metadata = []

    # --------------------------------------------
    # RUN RAG
    # --------------------------------------------

    for i, item in enumerate(
        source_dataset,
        start=1
    ):

        question = item["question"]

        print(
            f"[{i}/{len(source_dataset)}] "
            f"{question}"
        )

        try:

            result = run_rag(question)

        except Exception as e:

            print(
                f"\nRAG ERROR: {e}"
            )

            continue

        print("\nActual answer:")
        print(result["answer"])

        print(
            f"\nLatency: "
            f"{result['latency']:.2f}s"
        )

        # ----------------------------------------
        # RAGAS SAMPLE
        # ----------------------------------------

        sample = SingleTurnSample(
            user_input=question,

            response=result["answer"],

            retrieved_contexts=result["context"],

            reference=item["expected_answer"],
        )

        samples.append(sample)

        metadata.append({
            "id": item["id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "question": question,
            "expected_answer": item["expected_answer"],
            "actual_answer": result["answer"],
            "latency": result["latency"],
            "retrieval_scores": result["scores"],
        })

    # --------------------------------------------
    # CREATE RAGAS DATASET
    # --------------------------------------------

    evaluation_dataset = EvaluationDataset(
        samples=samples
    )

    # --------------------------------------------
    # METRICS
    # --------------------------------------------

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    response_relevancy = ResponseRelevancy(
        llm=evaluator_llm
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm
    )

    context_recall = ContextRecall(
        llm=evaluator_llm
    )

    metrics = [
        faithfulness,
        response_relevancy,
        context_precision,
        context_recall,
    ]

    # --------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------

    print("\n======================================")
    print("          RUNNING RAGAS")
    print("======================================\n")

    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
    )

    # --------------------------------------------
    # PRINT
    # --------------------------------------------

    print("\n======================================")
    print("          RAGAS RESULTS")
    print("======================================\n")

    print(results)

    # --------------------------------------------
    # CONVERT RESULTS
    # --------------------------------------------

    result_df = results.to_pandas()

    result_records = result_df.to_dict(
        orient="records"
    )

    # --------------------------------------------
    # ADD OUR METADATA
    # --------------------------------------------

    for i, record in enumerate(
        result_records
    ):

        if i < len(metadata):

            record.update(
                metadata[i]
            )

    # --------------------------------------------
    # SAVE
    # --------------------------------------------

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result_records,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        f"\nResults saved to:"
        f"\n{RESULT_PATH}"
    )


if __name__ == "__main__":
    main()