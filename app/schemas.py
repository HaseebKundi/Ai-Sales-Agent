from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    intent: str
    escalated: bool
    escalation_reason: Optional[str] = None
    confidence: float
    order_info: Optional[dict] = None


class TicketOut(BaseModel):
    id: int
    customer_message: str
    reason: str
    intent: Optional[str] = None
    order_id: Optional[str] = None
    status: str

    class Config:
        from_attributes = True
