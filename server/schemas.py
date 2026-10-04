from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(100, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(40, ge=1)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    text: str
