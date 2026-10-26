from langchain_ollama import ChatOllama
from tools.calculator import calculator
from langchain_core.messages import SystemMessage

from response import TutorResponse

class TutorAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2:1b",
            base_url="http://127.0.0.1:11434",
            temperature=0
        ) 

        self.llm_with_tools = self.llm.bind_tools(
            [calculator]
        )

        self.structured_llm = self.llm.with_structured_output(
            TutorResponse
        )


    def invoke(self,messages:str):

        system_message = SystemMessage(
            content="""
            You are a helpful study tutor.

            You have access to one tool:

            calculator:
            Use ONLY when the user asks you to perform
            a mathematical calculation.

            IMPORTANT:
            - Do NOT use calculator for general questions.
            - Do NOT use calculator for definitions.
            - Do NOT use calculator for explanations.
            - Do NOT use calculator for questions about AI, ML, RAG,
            Python, programming concepts, etc.
            - For normal knowledge questions, answer directly.
            - After receiving a calculator result, explain the result
            to the user.
            """)
        
        response = self.llm_with_tools.invoke(
            [system_message] + messages
        )

        return response

    def format_response(
            self,
            user_query,
            response:str,
            tool_used:bool,
            tool_name: str | None=None
    ):
        prompt = f"""
        Convert the tutor's response into the required structure.

        User question:
        {user_query}

        Tutor response:
        {response}

        Tool used:
        {tool_used}

        Tool name:
        {tool_name}
        """

        return self.structured_llm.invoke(prompt)

    
