import json
from pathlib import Path

import chromadb


# ==========================================================
# Configuration
# ==========================================================

RESULTS_FILE = "./evaluation/results.json"
OUTPUT_FILE = "./evaluation/results_with_context.json"

CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "company_documents"


# ==========================================================
# Load existing results
# ==========================================================

def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("results.json must contain a JSON object.")

    if "results" not in data:
        raise ValueError("results.json does not contain a 'results' field.")

    return data


# ==========================================================
# Connect to ChromaDB
# ==========================================================

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
    print(f"Total vectors in collection: {collection.count()}")

    return collection


# ==========================================================
# Add actual chunk text
# ==========================================================

def enrich_results(data, collection):
    results = data["results"]

    # Collect all chunk IDs from results.json
    all_chunk_ids = []

    for result in results:
        for chunk in result.get("retrieved_chunks", []):
            chunk_id = chunk.get("chunk_id")

            if chunk_id:
                all_chunk_ids.append(chunk_id)

    # Remove duplicates while preserving order
    all_chunk_ids = list(dict.fromkeys(all_chunk_ids))

    print(f"Unique retrieved chunks: {len(all_chunk_ids)}")

    # Chroma can retrieve multiple IDs at once.
    # Keep batches reasonably sized.
    batch_size = 500

    chunk_text_map = {}
    chunk_metadata_map = {}

    for start in range(0, len(all_chunk_ids), batch_size):

        batch_ids = all_chunk_ids[
            start:start + batch_size
        ]

        print(
            f"Fetching Chroma chunks "
            f"{start + 1}-{start + len(batch_ids)}..."
        )

        response = collection.get(
            ids=batch_ids,
            include=[
                "documents",
                "metadatas"
            ]
        )

        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])

        for index, chunk_id in enumerate(ids):

            text = None

            if index < len(documents):
                text = documents[index]

            metadata = None

            if index < len(metadatas):
                metadata = metadatas[index]

            chunk_text_map[chunk_id] = text
            chunk_metadata_map[chunk_id] = metadata

    # ------------------------------------------------------
    # Add text to every retrieved chunk
    # ------------------------------------------------------

    missing_ids = []

    for result in results:

        for chunk in result.get("retrieved_chunks", []):

            chunk_id = chunk.get("chunk_id")

            text = chunk_text_map.get(chunk_id)

            if text is None or not str(text).strip():

                missing_ids.append(chunk_id)

                continue

            chunk["text"] = str(text)

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    total_chunks = sum(
        len(result.get("retrieved_chunks", []))
        for result in results
    )

    chunks_with_text = sum(
        1
        for result in results
        for chunk in result.get("retrieved_chunks", [])
        if chunk.get("text")
    )

    print()
    print("=" * 70)
    print("CONTEXT ENRICHMENT SUMMARY")
    print("=" * 70)
    print(f"Total retrieved chunks : {total_chunks}")
    print(f"Chunks with text       : {chunks_with_text}")
    print(f"Missing chunks         : {len(set(missing_ids))}")

    if missing_ids:
        print()
        print("Missing chunk IDs:")

        for chunk_id in sorted(set(missing_ids)):
            print(f"  - {chunk_id}")

        raise RuntimeError(
            "Some retrieved chunks could not be found in ChromaDB. "
            "The output file was NOT created."
        )

    return data


# ==========================================================
# Save enriched results
# ==========================================================

def save_results(data):

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("RESULTS REGENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("RAG RESULTS CONTEXT ENRICHMENT")
    print("=" * 70)

    print("\nLoading existing results.json...")

    data = load_results()

    print(
        f"Questions found: "
        f"{len(data['results'])}"
    )

    print("\nConnecting to ChromaDB...")

    collection = get_collection()

    print("\nRetrieving actual chunk text...")

    data = enrich_results(
        data,
        collection
    )

    save_results(data)


if __name__ == "__main__":
    main()
