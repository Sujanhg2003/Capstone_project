from langchain_core.tools import tool
from typing import Literal

@tool
def calculate(
    a: float,
    b: float,
    operation: Literal[
        "add","subtract","multiply","divide"
    ]
) -> float:
    """
    Perform an arithmetic calculation on two numbers.

    Use:
    add for addition
    subtract for subtraction
    multiply for multiplication
    divide for division
    """

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            raise ValueError(
                "Cannot divide by zero"
            )

        return a / b

    else:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )