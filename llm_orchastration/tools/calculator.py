
import math


def calculate(
    a: float,
    b: float,
    operation: str,
) -> dict:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Inputs must be finite numbers.")

    if operation == "add":
        result = a + b

    elif operation == "subtract":
        result = a - b

    elif operation == "multiply":
        result = a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        result = a / b

    else:
        raise ValueError("Unsupported arithmetic operation.")

    if not math.isfinite(result):
        raise ValueError("Calculation produced a non-finite result.")

    return {"result": result}
