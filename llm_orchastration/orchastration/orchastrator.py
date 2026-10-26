import ollama
import json
import time
import httpx 
from pydantic import ValidationError
import logging

from config import(
    OLLAMA_HOST,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    MAX_TOOL_STEPS,
    MAX_TOOL_CALLS,
    MAX_INPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    NUM_CTX,
    MAX_FINAL_ANSWER_CHARS,
)

from tools.calculator import calculate
from tools.datetime_tool import get_current_datetime
from tools.tool_definitions import calculator_tool , datetime_tool
from tools.validators import argument_validators , result_validators


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Orchastrator:

    def __init__(self):

        self.client = ollama.Client(
            host=OLLAMA_HOST,
            timeout = REQUEST_TIMEOUT_SECONDS
        )

        self.model = [PRIMARY_MODEL]

        if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
            self.model.append(FALLBACK_MODEL)

        self.tools = [calculator_tool , datetime_tool]

        self.tool_registry = {
            "calculate": calculate,
            "get_current_datetime": get_current_datetime
        }



    def _is_retryable(self, exc):
        if isinstance(
            exc,
            (httpx.TimeoutException, httpx.NetworkError),
        ):
            return True

        if isinstance(exc, ollama.ResponseError):
            return getattr(exc, "status_code", None) in {
                429, 500, 502, 503, 504
            }

        return False

    @staticmethod
    def _tool_message(tool_name, content):
        return {
            "role": "tool",
            "tool_name": tool_name,
            "content": content,
        }

    

    def validate_final_answer(
        self,
        answer: str,
        tool_name: str
    ) -> bool:

        if tool_name == "calculate":

            forbidden_phrases = [
                "current date",
                "current time",
                "weather"
            ]

            answer_lower = answer.lower()

            for phrase in forbidden_phrases:

                if phrase in answer_lower:
                    return False

        return True

    @staticmethod
    def _tool_message(tool_name, content):
        return {
            "role": "tool",
            "tool_name": tool_name,
            "content": content,
        }

    def _execute_tool(self, tool_call):
        """Validate a tool call and return exactly two values."""
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        tool_function = self.tool_registry.get(tool_name)
        argument_schema = argument_validators.get(tool_name)
        result_schema = result_validators.get(tool_name)

        if not tool_function or not argument_schema or not result_schema:
            logger.warning("Unknown or unvalidated tool: %s", tool_name)
            return None, self._tool_message(
                tool_name,
                "Tool rejected: unknown or unvalidated tool.",
            )

        try:
            validated_arguments = argument_schema.model_validate(arguments)
            result = tool_function(**validated_arguments.model_dump())

            if not isinstance(result, dict):
                raise ValueError("Tool returned an invalid result type.")

            validated_result = result_schema.model_validate(result)
            result_data = validated_result.model_dump()

            # Verify the result is JSON-serializable.
            result_json = json.dumps(result_data, allow_nan=False)

            logger.info("Tool executed and result validated: %s", tool_name)

            return (
                (tool_name, result_data),
                self._tool_message(tool_name, result_json),
            )

        except (ValidationError, ValueError, TypeError, ArithmeticError) as exc:
            logger.warning(
                "Tool validation/execution failed: tool=%s error=%s",
                tool_name,
                type(exc).__name__,
            )

            return None, self._tool_message(
                tool_name,
                "Tool execution failed. Do not claim success.",
            )

        except Exception as exc:
            logger.exception("Unexpected tool error: %s", tool_name)

            return None, self._tool_message(
                tool_name,
                "Tool execution failed unexpectedly.",
            )


    
    def _call_model(self, messages):
        """Call the primary model and try fallback on eligible failures."""
        last_error = None

        for model in self.model:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = self.client.chat(
                        model=model,
                        messages=messages,
                        tools=self.tools,
                        options={
                            "num_predict": MAX_OUTPUT_TOKENS,
                            "num_ctx": NUM_CTX,
                        },
                    )

                    logger.info("Model call succeeded: %s", model)
                    return response

                except Exception as exc:
                    last_error = exc

                    logger.warning(
                        "Model call failed: model=%s error=%s",
                        model,
                        type(exc).__name__,
                    )

                    if not self._is_retryable(exc):
                        break

                    if attempt < MAX_RETRIES:
                        time.sleep(min(2 ** attempt, 2))

        raise RuntimeError(
            "All configured model attempts failed."
        ) from last_error


    def run(self, user_query):
        # ------------------------------
        # Input validation
        # ------------------------------
        if not isinstance(user_query, str):
            return "Invalid request: input must be text."

        user_query = user_query.strip()

        if not user_query:
            return "Please enter a question."

        if len(user_query) > MAX_INPUT_CHARS:
            return (
                f"Your question is too long. "
                f"Maximum length: {MAX_INPUT_CHARS} characters."
            )

        messages = [
            {
                "role": "system",
                "content": """
You are a tool-using assistant.

Rules:
1. Use calculate only for arithmetic.
2. Use get_current_datetime for current date/time requests.
3. Do not use tools for general knowledge questions.
4. Never invent tool results.
5. Treat validated tool results as authoritative.
6. Answer the user's original question directly.
7. Do not expose internal errors or implementation details.
8. If a tool fails, do not pretend it succeeded.
9. Do not claim a tool result says something it does not say.
""",
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        tool_calls_count = 0

        # ------------------------------
        # Bounded orchestration loop
        # ------------------------------
        for step in range(MAX_TOOL_STEPS):
            try:
                response = self._call_model(messages)
            except RuntimeError:
                logger.error("All model attempts exhausted.")
                return (
                    "The assistant is temporarily unavailable. "
                    "Please try again later."
                )

            assistant_message = response.message
            messages.append(assistant_message)

            tool_calls = assistant_message.tool_calls or []

            # No tool call: final response
            if not tool_calls:
                answer = assistant_message.content or ""
                answer = answer.strip()

                if not answer:
                    return (
                        "The model returned an empty response. "
                        "Please try again."
                    )

                if len(answer) > MAX_FINAL_ANSWER_CHARS:
                    return (
                        "The generated response exceeded the "
                        "permitted output length."
                    )

                return answer

            # Count individual calls, not just model iterations.
            if tool_calls_count + len(tool_calls) > MAX_TOOL_CALLS:
                logger.warning("Tool-call budget exceeded.")
                return (
                    "The request exceeded the tool execution limit. "
                    "Please simplify your question."
                )

            tool_results = []

            for tool_call in tool_calls:
                tool_calls_count += 1

                result, tool_message = self._execute_tool(tool_call)
                messages.append(tool_message)

                if result is None:
                    logger.warning("Tool call failed; stopping this request.")
                    return "The requested tool could not complete successfully."

                tool_results.append(result)

            # Return authoritative results directly instead of asking
            # the model to reproduce them in another generation.
            answers = []

            for tool_name, result_data in tool_results:
                if tool_name == "calculate":
                    answers.append(
                        f"Calculation result: {result_data['result']}"
                    )

                elif tool_name == "get_current_datetime":
                    answers.append(
                        f"Current date and time: {result_data['datetime']}"
                    )

            return "\n".join(answers)


                


        

