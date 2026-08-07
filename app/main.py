from typing import List

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.schemas import ChatRequest, ChatResponse, TicketOut
from app.agent import run_agent, run_agent_plan, llm
from app.models import SupportTicket
from app.seed_data import seed
from app.rag import build_or_load_vectorstore
from app import memory as agent_memory

app = FastAPI(
    title="AI Customer Support Agent",
    description="LangGraph agent with FAQ RAG, order lookup, streaming, memory, and human escalation.",
    version="1.1.0",
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
    build_or_load_vectorstore()  # warm up embeddings/FAQ index during startup


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_agent(req.message, req.session_id or "default")
    return ChatResponse(
        response=result["final_response"],
        intent=result["intent"],
        escalated=result["escalated"],
        escalation_reason=result.get("escalation_reason"),
        confidence=result.get("confidence", 0.0),
        order_info=result.get("order_info"),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    session_id = req.session_id or "default"
    plan = run_agent_plan(req.message, session_id)

    def event_generator():
        full_text = ""
        if plan.get("prompt_to_stream"):
            for chunk in llm.stream(plan["prompt_to_stream"]):
                token = chunk.content or ""
                if token:
                    full_text += token
                    yield token
        else:
            full_text = plan.get("final_response", "")
            yield full_text
        agent_memory.add_exchange(session_id, req.message, full_text)

    headers = {
        "X-Intent": plan.get("intent", ""),
        "X-Escalated": str(plan.get("escalated", False)),
        "X-Confidence": str(plan.get("confidence", 0.0)),
    }
    return StreamingResponse(event_generator(), media_type="text/plain", headers=headers)


@app.get("/tickets", response_model=List[TicketOut])
def list_tickets(db: Session = Depends(get_db)):
    return db.query(SupportTicket).order_by(SupportTicket.id.desc()).all()


app.mount("/", StaticFiles(directory="static", html=True), name="static")
