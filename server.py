from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"


class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    response: str

@app.post("/generate" , response_model=PromptResponse)
def generate_text(request: PromptRequest):

    payload = {
        "model": MODEL,
        "prompt": request.prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    response.raise_for_status()

    data = response.json()

    return {
        "response": data["response"]
    }