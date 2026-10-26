from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

from config import (
    PDF_DIR ,
    CHUNK_SIZE ,
    CHUNK_OVERLAP , 
    OLLAMA_BASE_URL ,
    EMBEDDING_MODEL , 
    CHROMA_DIR ,
    COLLECTION_NAME
    )

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def extract_fir_metadata(filename: str) -> dict:

    name = filename.replace(".pdf", "")

    match = re.match(
        r"FIR_(\d+)_([^\.]+)",
        name
    )

    if not match:
        return {
            "fir_id": "unknown",
            "case_type": "unknown"
        }

    fir_id = match.group(1)

    case_type = match.group(2).replace("_", " ")

    return {
        "fir_id": f"SYN-FIR-{int(fir_id):03d}",
        "case_type": case_type
    }

def load_fir_documents():

    documents = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        fir_metadata = extract_fir_metadata(
            pdf_file.name
        )

        print(
            f"  FIR ID    : {fir_metadata['fir_id']}"
        )

        print(
            f"  Case Type : {fir_metadata['case_type']}"
        )

        loader = PyPDFLoader(
            str(pdf_file)
        )

        docs = loader.load()

        for doc in docs:

            doc.metadata.update({

                "source_file": pdf_file.name,

                "document_type": "FIR",

                "fir_id": fir_metadata["fir_id"],

                "case_type": fir_metadata["case_type"],

            })

        documents.extend(docs)

    print(
        f"\nTotal pages loaded: {len(documents)}"
    )

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= CHUNK_SIZE,
        chunk_overlap= CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)
    print(f"Total chunks:{len(chunks)}")

    return chunks


def create_vectorstore(chunks):

    print("Loading Ollama embedding model...")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url= OLLAMA_BASE_URL
    )

    print("Creating ChromaDB...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print("Adding documents to ChromaDB...")

    vectorstore.add_documents(chunks)

    print("Documents successfully stored.")

    return vectorstore


if __name__ == "__main__":

    documents = load_fir_documents()

    chunks = split_documents(documents)

    create_vectorstore(chunks)

    print("\n========== INGESTION COMPLETE ==========")