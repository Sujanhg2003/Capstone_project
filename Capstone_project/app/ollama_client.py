from ollama import Client

from app.config import (
    OLLAMA_HOST , LLM_MODEL , TEMPERATURE
)

class OllamaLLM:

    def __init__(self):

        self.client = Client(
            host=OLLAMA_HOST
        )

        self.model = LLM_MODEL

    def generate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            think= False,
            options={
                "temperature": TEMPERATURE
            }
        )

        return response.message.content