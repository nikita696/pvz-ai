from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=12000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    provider: str
    model: str
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
