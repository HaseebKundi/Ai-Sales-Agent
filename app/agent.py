import re
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Order, SupportTicket
from app.rag import search_faq
from app import memory

llm = ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL, temperature=0)

ORDER_ID_PATTERN = re.compile(r"ord-?\s?(\d{3,})", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b(\d{3,})\b")


def extract_order_id(text: str):
    match = ORDER_ID_PATTERN.search(text)
    if match:
        return f"ORD-{match.group(1)}"
    match = NUMBER_PATTERN.search(text)
    if match:
        return f"ORD-{match.group(1)}"
    return None


class AgentState(TypedDict):
    user_message: str
    session_id: str
    chat_history: str
    intent: str
    order_id: Optional[str]
    order_info: Optional[dict]
    order_lookup_status: Optional[str]  # "found" | "not_found" | "missing_id"
    faq_matches: Optional[List[dict]]
    confidence: float
    prompt_to_stream: Optional[str]
    final_response: str
    escalated: bool
    escalation_reason: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def classify_intent(state: AgentState) -> AgentState:
    history = memory.format_history_for_prompt(state["session_id"])
    prompt = f"""You are an intent classifier for an e-commerce customer support bot.
Classify the user's LATEST message into exactly one of these categories:
- "order_status": asking about an order, shipment, tracking, delivery, or refund status of a SPECIFIC order
- "faq": general policy questions (returns, shipping times, payment methods, cancellations) not tied to a specific order
- "complaint": user is frustrated, angry, or reporting a problem that needs a human (wrong item, damaged product, billing dispute)
- "chitchat": greetings, thanks, acknowledgements, compliments, casual small talk, questions about the assistant itself
  (e.g. "who are you"), or questions about the CONVERSATION SO FAR (e.g. "what's my name", "what did I just say",
  "can you repeat that") — anything answerable just from being friendly or from re-reading the chat history
- "other": unclear or off-topic requests that genuinely don't fit any category above (e.g. asking something
  completely unrelated to shopping/support that also isn't casual conversation, like asking for a recipe)

Recent conversation (for context only):
{history}

Latest customer message: "{state['user_message']}"

Respond with ONLY the category word, nothing else.
"""
    result = llm.invoke(prompt)
    intent = result.content.strip().lower()
    if intent not in ("order_status", "faq", "complaint", "chitchat", "other"):
        intent = "other"

    order_id = extract_order_id(state["user_message"])

    return {**state, "intent": intent, "order_id": order_id, "chat_history": history}


def lookup_order(state: AgentState) -> AgentState:
    order_id = state.get("order_id")
    if not order_id:
        return {**state, "order_info": None, "order_lookup_status": "missing_id", "confidence": 1.0}

    db: Session = SessionLocal()
    try:
        order = db.query(Order).filter(Order.order_id == order_id).first()
    finally:
        db.close()

    if not order:
        return {**state, "order_info": None, "order_lookup_status": "not_found", "confidence": 1.0}

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
    return {**state, "order_info": order_info, "order_lookup_status": "found", "confidence": 0.95}


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
    return state


def build_order_prompt(state: AgentState) -> str:
    status = state.get("order_lookup_status")
    history = state.get("chat_history", "")

    if status == "found":
        import json
        return f"""You are a friendly customer support assistant. Using ONLY this order
data, answer the customer's question naturally in 2-4 sentences. Do not invent details.

Recent conversation (for context):
{history}

Order data: {json.dumps(state["order_info"])}
Customer question: "{state['user_message']}"
"""
    if status == "not_found":
        return f"""You are a friendly customer support assistant. The customer asked about
order ID "{state.get('order_id')}", but no order with that ID exists in our system.
Politely tell them this order number doesn't exist, ask them to double-check the ID,
and offer to connect them with a human if they still need help. Keep it to 2-3 sentences.

Recent conversation (for context):
{history}

Customer message: "{state['user_message']}"
"""
    return f"""You are a friendly customer support assistant. The customer is asking about
an order but didn't include an order ID. Politely ask them to share their order ID
(e.g. ORD-1001) so you can look it up. Keep it to 1-2 sentences.

Recent conversation (for context):
{history}

Customer message: "{state['user_message']}"
"""


def build_faq_prompt(state: AgentState) -> str:
    history = state.get("chat_history", "")
    matches = state.get("faq_matches") or []
    context = "\n\n".join(f"Q: {m['question']}\nA: {m['answer']}" for m in matches)
    return f"""You are a friendly customer support assistant. Using ONLY the FAQ context
below, answer the customer's question naturally in 2-4 sentences.

Recent conversation (for context):
{history}

FAQ context:
{context}

Customer question: "{state['user_message']}"
"""


def build_chitchat_prompt(state: AgentState) -> str:
    history = state.get("chat_history", "")
    return f"""You are a friendly customer support assistant. The customer just sent a
casual message — a greeting, thanks, small talk, or a question about the conversation
itself (like their name, or something they said earlier).

If the answer is present in the recent conversation below, use it and answer directly
and confidently (e.g. if they told you their name earlier, use it now). If it genuinely
isn't in the conversation, say so honestly and briefly — don't guess or make something up.

Keep your reply natural and warm, 1-2 sentences. If there's nothing specific to answer,
invite them to ask about an order or a store policy if they need anything.

Recent conversation (for context):
{history}

Customer message: "{state['user_message']}"
"""


def answer_directly(state: AgentState) -> AgentState:
    if state["intent"] == "order_status":
        prompt = build_order_prompt(state)
    else:
        prompt = build_faq_prompt(state)
    return {**state, "prompt_to_stream": prompt, "final_response": "", "escalated": False, "escalation_reason": None}


def answer_chitchat(state: AgentState) -> AgentState:
    prompt = build_chitchat_prompt(state)
    return {**state, "prompt_to_stream": prompt, "final_response": "", "escalated": False, "escalation_reason": None}


def escalate_to_human(state: AgentState) -> AgentState:
    if state["intent"] == "complaint":
        reason = "Customer complaint requiring human judgment."
    elif state["intent"] == "faq" and state.get("confidence", 0) < settings.CONFIDENCE_THRESHOLD:
        reason = "No confident FAQ match found."
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
    return {**state, "prompt_to_stream": None, "final_response": response, "escalated": True, "escalation_reason": reason}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_classify(state: AgentState) -> str:
    if state["intent"] == "order_status":
        return "lookup_order"
    if state["intent"] == "faq":
        return "check_faq"
    if state["intent"] == "chitchat":
        return "chitchat"
    return "escalate"


def route_after_confidence(state: AgentState) -> str:
    if state["intent"] == "order_status":
        return "answer_directly"
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


def _initial_state(user_message: str, session_id: str) -> AgentState:
    return {
        "user_message": user_message,
        "session_id": session_id,
        "chat_history": "",
        "intent": "",
        "order_id": None,
        "order_info": None,
        "order_lookup_status": None,
        "faq_matches": None,
        "confidence": 0.0,
        "prompt_to_stream": None,
        "final_response": "",
        "escalated": False,
        "escalation_reason": None,
    }


def run_agent(user_message: str, session_id: str = "default") -> AgentState:
    """Non-streaming path: runs the graph, generates the answer in one shot,
    and saves the exchange to memory. Used by /chat."""
    result = agent.invoke(_initial_state(user_message, session_id))
    if result.get("prompt_to_stream"):
        llm_result = llm.invoke(result["prompt_to_stream"])
        result["final_response"] = llm_result.content.strip()
    memory.add_exchange(session_id, user_message, result["final_response"])
    return result


def run_agent_plan(user_message: str, session_id: str = "default") -> AgentState:
    """Runs the graph up to the point of deciding WHAT to say, without
    generating the text yet. Used by /chat/stream, which then streams the
    LLM's answer token-by-token using result['prompt_to_stream']."""
    return agent.invoke(_initial_state(user_message, session_id))
