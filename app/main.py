from typing import List

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.schemas import ChatRequest, ChatResponse, TicketOut
from app.agent import run_agent
from app.models import SupportTicket
from app.seed_data import seed

app = FastAPI(
    title="AI Customer Support Agent",
    description="LangGraph agent with FAQ RAG, order lookup, and human escalation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    seed()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message)
    return ChatResponse(
        response=result["final_response"],
        intent=result["intent"],
        escalated=result["escalated"],
        escalation_reason=result.get("escalation_reason"),
        confidence=result.get("confidence", 0.0),
        order_info=result.get("order_info"),
    )


@app.get("/tickets", response_model=List[TicketOut])
def list_tickets(db: Session = Depends(get_db)):
    """Lists all support tickets created by the escalation node — this is
    the 'inbox' a human agent would work from."""
    return db.query(SupportTicket).order_by(SupportTicket.id.desc()).all()


# Chat widget UI (index.html) served at "/" — mounted last so it never
# shadows the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
