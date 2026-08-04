import json
import re
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Order, SupportTicket
from app.rag import search_faq

llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)

ORDER_ID_PATTERN = re.compile(r"ord-?\s?(\d{3,})", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b(\d{3,})\b")


def extract_order_id(text: str):
    """Finds an order number in flexible formats: 'ORD-1001', 'ord1001',
    'order 1001', '#1001', or just '1001' on its own."""
    match = ORDER_ID_PATTERN.search(text)
    if match:
        return f"ORD-{match.group(1)}"
    match = NUMBER_PATTERN.search(text)
    if match:
        return f"ORD-{match.group(1)}"
    return None


# ---------------------------------------------------------------------------
# Shared state that flows through every node in the graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    user_message: str
    intent: str
    order_id: Optional[str]
    order_info: Optional[dict]
    faq_matches: Optional[List[dict]]
    confidence: float
    final_response: str
    escalated: bool
    escalation_reason: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def classify_intent(state: AgentState) -> AgentState:
    prompt = f"""You are an intent classifier for an e-commerce customer support bot.
Classify the user's message into exactly one of these categories:
- "order_status": asking about an order, shipment, tracking, delivery, or refund status of a SPECIFIC order
- "faq": general policy questions (returns, shipping times, payment methods, cancellations) not tied to a specific order
- "complaint": user is frustrated, angry, or reporting a problem that needs a human (wrong item, damaged product, billing dispute)
- "chitchat": greetings, thanks, acknowledgements, compliments, or casual small talk (e.g. "good", "thanks", "hi", "you're helpful")
- "other": unclear or off-topic requests that don't fit any category above

Respond with ONLY the category word, nothing else.

User message: "{state['user_message']}"
"""
    result = llm.invoke(prompt)
    intent = result.content.strip().lower()
    if intent not in ("order_status", "faq", "complaint", "chitchat", "other"):
        intent = "other"

    order_id = extract_order_id(state["user_message"])

    return {**state, "intent": intent, "order_id": order_id}


def lookup_order(state: AgentState) -> AgentState:
    order_id = state.get("order_id")
    if not order_id:
        # They asked about an order but gave no ID -> can't help confidently
        return {**state, "order_info": None, "confidence": 0.2}

    db: Session = SessionLocal()
    try:
        order = db.query(Order).filter(Order.order_id == order_id).first()
    finally:
        db.close()

    if not order:
        return {**state, "order_info": None, "confidence": 0.2}

    order_info = {
        "order_id": order.order_id,
        "item_name": order.item_name,
        "status": order.status,
        "tracking_number": order.tracking_number,
        "carrier": order.carrier,
        "estimated_delivery": (
            order.estimated_delivery.strftime("%Y-%m-%d") if order.estimated_delivery else None
        ),
        "total_amount": order.total_amount,
    }
    return {**state, "order_info": order_info, "confidence": 0.95}


def check_faq(state: AgentState) -> AgentState:
    results = search_faq(state["user_message"], k=2)
    if not results:
        return {**state, "faq_matches": [], "confidence": 0.0}

    matches = [
        {"question": doc.metadata["question"], "answer": doc.metadata["answer"], "score": score}
        for doc, score in results
    ]
    return {**state, "faq_matches": matches, "confidence": matches[0]["score"]}


def decide_confidence(state: AgentState) -> AgentState:
    # Pass-through node; the actual routing decision happens in the
    # conditional edge below. Kept as its own node so the graph makes the
    # "confidence gate" step explicit and easy to reason about / extend.
    return state

def answer_chitchat(state: AgentState) -> AgentState:
    prompt = f"""You are a friendly customer support assistant. The customer just
sent a casual message (greeting, thanks, or small talk) rather than a support
question. Reply naturally and warmly in 1-2 sentences, and invite them to ask
about an order or a store policy if they need anything.

Customer message: "{state['user_message']}"
"""
    result = llm.invoke(prompt)
    return {**state, "final_response": result.content.strip(), "escalated": False, "escalation_reason": None}


def answer_directly(state: AgentState) -> AgentState:
    if state["intent"] == "order_status" and state.get("order_info"):
        prompt = f"""You are a friendly customer support assistant. Using ONLY this order
data, answer the customer's question naturally in 2-4 sentences. Do not invent details.

Order data: {json.dumps(state["order_info"])}
Customer question: "{state['user_message']}"
"""
    else:
        matches = state.get("faq_matches") or []
        context = "\n\n".join(f"Q: {m['question']}\nA: {m['answer']}" for m in matches)
        prompt = f"""You are a friendly customer support assistant. Using ONLY the FAQ context
below, answer the customer's question naturally in 2-4 sentences.

FAQ context:
{context}

Customer question: "{state['user_message']}"
"""
    result = llm.invoke(prompt)
    return {**state, "final_response": result.content.strip(), "escalated": False, "escalation_reason": None}


def escalate_to_human(state: AgentState) -> AgentState:
    if state["intent"] == "order_status" and not state.get("order_info"):
        reason = "Order ID missing or not found in system."
    elif state["intent"] == "faq" and state.get("confidence", 0) < settings.CONFIDENCE_THRESHOLD:
        reason = "No confident FAQ match found."
    elif state["intent"] == "complaint":
        reason = "Customer complaint requiring human judgment."
    else:
        reason = "Message did not match a known support intent."

    db: Session = SessionLocal()
    try:
        db.add(
            SupportTicket(
                customer_message=state["user_message"],
                reason=reason,
                intent=state["intent"],
                order_id=state.get("order_id"),
            )
        )
        db.commit()
    finally:
        db.close()

    response = (
        "I want to make sure you get the right help here, so I'm connecting you with a "
        "member of our support team. They'll follow up on this shortly. Thanks for your patience!"
    )
    return {**state, "final_response": response, "escalated": True, "escalation_reason": reason}


# ---------------------------------------------------------------------------
# Routing (conditional edges)
# ---------------------------------------------------------------------------
def route_after_classify(state: AgentState) -> str:
    if state["intent"] == "order_status":
        return "lookup_order"
    if state["intent"] == "faq":
        return "check_faq"
    if state["intent"] == "chitchat":
        return "chitchat"
    return "escalate"  # complaint / other -> straight to a human


def route_after_confidence(state: AgentState) -> str:
    if state["confidence"] >= settings.CONFIDENCE_THRESHOLD:
        return "answer_directly"
    return "escalate"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("lookup_order", lookup_order)
    graph.add_node("check_faq", check_faq)
    graph.add_node("decide_confidence", decide_confidence)
    graph.add_node("answer_directly", answer_directly)
    graph.add_node("chitchat", answer_chitchat)
    graph.add_node("escalate", escalate_to_human)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "lookup_order": "lookup_order",
            "check_faq": "check_faq",
            "chitchat": "chitchat",
            "escalate": "escalate",
        },
    )

    graph.add_edge("chitchat", END)
    graph.add_edge("lookup_order", "decide_confidence")
    graph.add_edge("check_faq", "decide_confidence")
    graph.add_conditional_edges(
        "decide_confidence",
        route_after_confidence,
        {"answer_directly": "answer_directly", "escalate": "escalate"},
    )
    graph.add_edge("answer_directly", END)
    graph.add_edge("escalate", END)

    return graph.compile()


agent = build_agent()


def run_agent(user_message: str) -> AgentState:
    initial_state: AgentState = {
        "user_message": user_message,
        "intent": "",
        "order_id": None,
        "order_info": None,
        "faq_matches": None,
        "confidence": 0.0,
        "final_response": "",
        "escalated": False,
        "escalation_reason": None,
    }
    return agent.invoke(initial_state)