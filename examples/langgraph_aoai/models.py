from pydantic import BaseModel, Field
from typing import Literal

class UserMessage(BaseModel):
    text: str = Field(..., description="Raw user input")

class RouteDecision(BaseModel):
    route: Literal["rag", "answer"]

class FinalAnswer(BaseModel):
    text: str

class SummaryResult(BaseModel):
    summary: str
