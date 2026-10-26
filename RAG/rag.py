from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from retrieve import retrieve_documents

from config import (
    PDF_DIR ,
    CHUNK_SIZE ,
    CHUNK_OVERLAP , 
    OLLAMA_BASE_URL ,
    EMBEDDING_MODEL , 
    CHROMA_DIR ,
    COLLECTION_NAME,
    LLM_MODEL
    )

llm = ChatOllama(
    model= LLM_MODEL , 
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
    You are an FIR analysis assistant.

    Answer the user's question ONLY using the FIR context provided below.

    Rules:
    1. Do not invent information.
    2. Do not use outside knowledge.
    3. If the answer is not present in the context, say:
    "I could not find this information in the provided FIR records."
    4. Mention the source FIR file whenever possible.
    5. Be concise and factual.

    FIR CONTEXT:
    ----------------
    {context}
    ----------------

    USER QUESTION:
    {question}

    ANSWER:
    """)

def build_context(documents):

    context_parts = []

    for i , doc in enumerate(documents ,start=1):
        source = doc.metadata.get(
            "source_file",
            "Unknown source"
        )

        context_parts.append(
            f"""
SOURCE {i}: {source}

{doc.page_content}
"""
        )

    return "\n".join(context_parts)

def ask_fir_question(question):

    # 1. Retrieve relevant documents
    documents , scores = retrieve_documents(question)

    # 2. Build context
    context = build_context(documents)

    # 3. Create prompt
    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    # 4. Generate answer
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "context": [
            doc.page_content
            for doc in documents
        ],
        "scores": scores
    }


if __name__ == "__main__":
    while True:

        question = input("\nAsk your FIR question: ")
        answer = ask_fir_question(question)

        print("\n========== ANSWER ==========\n")
        print(answer["answer"])