from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    TOP_K,
)


def get_vectorstore():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    return vectorstore

def retrieve_documents(query):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query,
        k=TOP_K
    )

    documents = [doc for doc, score in results]

    scores = [score for doc, score in results]

    return documents, scores

if __name__ == "__main__":

    documents , scores = retrieve_documents(
    "Find FIRs related to mobile phone theft"
)

    for i, (doc, score) in enumerate(zip(documents,scores), start=1):

        print(f"\nResult {i}")

        print("Source:")
        print(doc.metadata["source_file"])

        print("Case:")
        print(doc.metadata["case_type"])

        print("Score:")
        print(score)

        print("Content:")
        print(doc.page_content)

        print("-" * 80)