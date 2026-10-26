from pydantic import BaseModel , Field
from langchain_ollama import ChatOllama

class VerificationResult(BaseModel):
    valid:bool = Field(
        description="Whether the tutor response is valid"
    )

    reason: str = Field(
        description="Reason for the validation result"
    )

class VerifierAgent:

    def __init__(self):

        self.llm = ChatOllama(
            model="llama3.2:1b",
            base_url="http://127.0.0.1:11434",
            temperature=0
        )

        self.structured_llm = self.llm.with_structured_output(
            VerificationResult
        )

    def verify(self,user_query:str,tutor_response:str):
        prompt = f"""
        You are a response verification agent.

        Check whether the tutor's response correctly answers
        the user's question.

        User question:
        {user_query}

        Tutor response:
        {tutor_response}

        Determine whether the response is valid.

        Return:
        - valid = true if the response is acceptable
        - valid = false if the response is incorrect
        - reason = short explanation
        """

        return self.structured_llm.invoke(prompt)