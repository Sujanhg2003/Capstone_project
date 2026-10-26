import json
import re
from pathlib import Path

from ollama import Client

from app.config import (
    OLLAMA_HOST,
    LLM_MODEL
)

from app.ingestion.document_loader import (
    load_all_pdfs
)


DOCUMENT_DIRECTORY = "./data/documents"

OUTPUT_FILE = (
    "./evaluation/dataset.json"
)

QUESTIONS_PER_PAGE = 2


client = Client(
    host=OLLAMA_HOST
)


# ==========================================================
# Generate questions from one page
# ==========================================================

def generate_questions(
    document_name: str,
    page_number: int,
    page_text: str
) -> list[dict]:

    prompt = f"""
You are creating an evaluation dataset for an
internal company RAG system.

Use ONLY the provided document content.

DOCUMENT:
{document_name}

PAGE:
{page_number}

CONTENT:
{page_text}

Generate exactly {QUESTIONS_PER_PAGE}
question-answer pairs that can be answered
directly from this page.

Rules:

1. Questions must be answerable using ONLY
   the provided content.

2. Answers must contain ONLY information
   supported by the content.

3. Do not use outside knowledge.

4. Do not invent names, dates, numbers,
   policies, or facts.

5. Questions should be realistic questions
   an employee might ask.

6. Prefer factual questions that test
   important information.

7. Return ONLY valid JSON.

Required format:

[
    {{
        "question": "...",
        "ground_truth": "..."
    }}
]
"""

    response = client.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You generate factual evaluation "
                    "datasets from documents. "
                    "Never use outside knowledge."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        think=False,

        options={
            "temperature": 0.0,
            "num_predict": 500
        }
    )

    content = response.message.content.strip()

    # ------------------------------------------------------
    # Remove markdown code fences if model adds them
    # ------------------------------------------------------

    content = re.sub(
        r"```json\s*",
        "",
        content
    )

    content = re.sub(
        r"```\s*$",
        "",
        content
    )

    content = content.strip()

    # ------------------------------------------------------
    # Parse JSON
    # ------------------------------------------------------

    try:

        data = json.loads(
            content
        )

    except json.JSONDecodeError as error:

        print(
            f"\nJSON parsing failed "
            f"for {document_name}, "
            f"page {page_number}"
        )

        print(
            "Model output:"
        )

        print(content)

        print(
            f"\nError: {error}"
        )

        return []

    # ------------------------------------------------------
    # Validate structure
    # ------------------------------------------------------

    if not isinstance(
        data,
        list
    ):
        print(
            "Invalid response: "
            "expected a list."
        )

        return []

    valid_questions = []

    for item in data:

        if not isinstance(
            item,
            dict
        ):
            continue

        question = item.get(
            "question"
        )

        ground_truth = item.get(
            "ground_truth"
        )

        if not question or not ground_truth:
            continue

        valid_questions.append({

            "question": question.strip(),

            "ground_truth": (
                ground_truth.strip()
            )
        })

    return valid_questions


# ==========================================================
# Main dataset generation
# ==========================================================

def main():

    print("=" * 70)
    print("RAG EVALUATION DATASET GENERATOR")
    print("=" * 70)

    # ------------------------------------------------------
    # Create evaluation directory
    # ------------------------------------------------------

    Path(
        "./evaluation"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------
    # Load PDFs
    # ------------------------------------------------------

    print(
        "\nLoading PDF documents..."
    )

    documents = load_all_pdfs(
        DOCUMENT_DIRECTORY
    )

    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    dataset = []

    question_id = 1

    # ------------------------------------------------------
    # Process each document
    # ------------------------------------------------------

    for document in documents:

        file_name = document[
            "file_name"
        ]

        print(
            f"\nProcessing: {file_name}"
        )

        # --------------------------------------------------
        # Process each page
        # --------------------------------------------------

        for page in document[
            "pages"
        ]:

            page_number = page[
                "page_number"
            ]

            page_text = page[
                "text"
            ].strip()

            # Skip empty pages

            if not page_text:
                continue

            print(
                f"  Page {page_number}..."
            )

            questions = generate_questions(

                document_name=file_name,

                page_number=page_number,

                page_text=page_text
            )

            # ------------------------------------------------
            # Add source information
            # ------------------------------------------------

            for item in questions:

                record = {

                    "id": (
                        f"q{question_id:04d}"
                    ),

                    "question": item[
                        "question"
                    ],

                    "ground_truth": item[
                        "ground_truth"
                    ],

                    "expected_sources": [

                        {
                            "document": file_name,

                            "page": page_number
                        }

                    ],

                    "type": "answerable"
                }

                dataset.append(
                    record
                )

                question_id += 1

    # ------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)

    print(
        "DATASET GENERATION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Total questions: "
        f"{len(dataset)}"
    )

    print(
        f"Output file: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()