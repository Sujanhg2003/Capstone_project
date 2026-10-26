import json
from pathlib import Path
from datetime import datetime , timezone

from app.rag.simple_pipeline import SimpleRAGPipeline

DATASET_FILE = Path(
    "./evaluation/dataset.json"
)

RESULTS_FILE = Path(
    "./evaluation/results.json"
)

def load_dataset():

    with DATASET_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)

def save_results(results):

    evaluation_output = {

        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "total_questions": len(results),

        "results": results
    }

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_output,
            file,
            indent=4,
            ensure_ascii=False
        )

def main():

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    print(
        f"\nEvaluation questions: "
        f"{len(dataset)}"
    )

    rag = SimpleRAGPipeline()

    results = []

    for index, item in enumerate(
        dataset,
        start=1
    ):

        question_id = item["id"]
        question = item["question"]

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{question_id}"
        )

        print(
            f"Question: {question}"
        )

        try:

            result = rag.ask(
                question
            )

            evaluation_result = {

                "id": question_id,

                "question": question,

                "ground_truth": (
                    item.get(
                        "ground_truth"
                    )
                ),

                "expected_sources": (
                    item.get(
                        "expected_sources",
                        []
                    )
                ),

                "type": item.get(
                    "type",
                    "answerable"
                ),

                "answer": result[
                    "answer"
                ],

                "actual_sources": result[
                    "sources"
                ],

                "retrieved_chunks": [

                    {
                        "chunk_id": chunk[
                            "chunk_id"
                        ],

                        "distance": chunk[
                            "distance"
                        ],

                        "document": chunk[
                            "metadata"
                        ]["source"],

                        "page": chunk[
                            "metadata"
                        ]["page"]
                    }

                    for chunk in result[
                        "results"
                    ]
                ],

                "retrieval": {

                    "status": (
                        result[
                            "latency"
                        ].get(
                            "retrieval_status"
                        )
                    ),

                    "chunks_returned": len(
                        result[
                            "results"
                        ]
                    )
                },

                "latency": result[
                    "latency"
                ],

                "request_id": result[
                    "request_id"
                ]
            }

            results.append(
                evaluation_result
            )

            print(
                "Status: completed"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            results.append({

                "id": question_id,

                "question": question,

                "error": str(error)
            })

    save_results(
        results
    )

    print("\n" + "=" * 70)

    print(
        "EVALUATION RUN COMPLETED"
    )

    print("=" * 70)

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()