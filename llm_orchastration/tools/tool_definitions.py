
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Perform arithmetic on two explicitly provided numbers. "
            "Use only for addition, subtraction, multiplication, "
            "or division. Provide a, b, and operation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First operand.",
                },
                "b": {
                    "type": "number",
                    "description": "Second operand.",
                },
                "operation": {
                    "type": "string",
                    "enum": [
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                    ],
                },
            },
            "required": ["a", "b", "operation"],
        },
    },
}

datetime_tool = {
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description": (
            "Get the current local date and time, including timezone "
            "offset. Use for requests asking for the current date or time."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
