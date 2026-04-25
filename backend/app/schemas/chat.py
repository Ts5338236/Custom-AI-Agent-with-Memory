from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = Field(..., min_length=1, max_length=100)

class ChatResponse(BaseModel):
    response: str
    session_id: str
