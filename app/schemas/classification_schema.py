from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ClassificationRequest(BaseModel):
    message: NonBlankString = Field(description="Customer support message")


class ClassificationResponse(BaseModel):
    category: Literal[
        "Billing",
        "Technical Support",
        "Account Issues",
        "General Inquiry",
    ]
    probability: float = Field(ge=0, le=1, description="Model-estimated confidence")
    reason: NonBlankString = Field(description="Brief justification for the category")
