from typing import Optional

from langchain_ollama import ChatOllama

class OllamaJudge(DeepEvalBaseLLM):

    def __init__(
            self,
            model = "qwen3:8b",
            base_url="http://localhost:11434",
    ):
        self.model_name = model
        self.llm = ChatOllama(
            model = model,
            base_url=base_url,
            temperature=0,
        )

    def load_model(self):
        return self.llm

    def generate(
            self,
            prompt: str,
            schema: Optional[object] = None,
    ):

        response = self.llm.invoke(prompt)
        return response.content

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[object] = None,
    ):

        response = await self.llm.ainvoke(prompt)

        return response.content

    def get_model_name(self):

        return self.model_name