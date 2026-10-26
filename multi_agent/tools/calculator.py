from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Use this tool when an exact mathematical calculation is required.
    """

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"