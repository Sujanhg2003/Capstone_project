from ollama import Client

from app.config import (
    OLLAMA_HOST,EMBEDDING_MODEL
)

class OllamaEmbedder:

    def __init__(self):
        self.client = Client(host = OLLAMA_HOST)
        self.model = EMBEDDING_MODEL

    def embed_text(self,text:str) ->list[float]:

        response = self.client.embed(
            model= self.model,
            input=text
        )

        return response.embeddings[0]

    def embed_documents(
            self,
            texts: list[str]
    ) -> list[list[float]]:

        response = self.client.embed(
            model=self.model,
            input=texts
        )

        return response.embeddings

    