from pydantic import BaseModel,Field

class TutorResponse(BaseModel):

    answer: str = Field(
        description="The final answer to the user's question"
    )

    tool_used: bool = Field(
        description="Whether a tool was used"
    )

    tool_name: str | None = Field(
        default=None,
        description="Name of the tool used, if any"
    )

class VerificationResult(BaseModel):

    valid: bool = Field(
        description="Whether the tutor response is correct"
    )

    reason: str = Field(
        description="Explanation for the validation result"
    )