from pydantic import BaseModel, field_validator

class AnswerPayload(BaseModel):
    text: str
    @field_validator("text")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("empty answer")
        return v
    
# TODO: Use after compose_node. payload = AnswerPayload(text=state["final"].text)