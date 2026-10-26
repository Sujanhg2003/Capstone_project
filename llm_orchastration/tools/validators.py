from typing import Literal
from pydantic import BaseModel, ConfigDict , Field , field_validator

class CalculatorArguments(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    a: float = Field(allow_inf_nan=False)
    b: float = Field(allow_inf_nan=False)

    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide"
    ]


class DateTimeArguments(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )


argument_validators = {
    "calculate": CalculatorArguments,
    "get_current_datetime": DateTimeArguments
}

class CalculatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: float = Field(allow_inf_nan=False)


class DateTimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datetime: str

    @field_validator("datetime")
    @classmethod
    def validate_datetime(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Datetime cannot be empty.")
        return value

result_validators = {
    "calculate": CalculatorResult,
    "get_current_datetime": DateTimeResult
}