from typing import Any

from langchain_ollama import ChatOllama

from ragas.llms import LangchainLLMWrapper

def get_evaluator_llm():

    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    return LangchainLLMWrapper(llm)