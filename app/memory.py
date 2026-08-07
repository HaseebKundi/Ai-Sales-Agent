"""Simple in-memory conversation history, keyed by session_id.

Note: this resets whenever the server restarts (redeploys, or Render's free
tier spinning down after inactivity). For memory that survives restarts,
swap this for a database table keyed by session_id instead.
"""

from typing import List, Dict

MAX_TURNS_KEPT = 6  # how many past user/assistant exchanges to keep per session

_conversations: Dict[str, List[dict]] = {}


def get_history(session_id: str) -> List[dict]:
    return _conversations.get(session_id, [])


def add_exchange(session_id: str, user_message: str, assistant_message: str):
    history = _conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    max_messages = MAX_TURNS_KEPT * 2
    if len(history) > max_messages:
        _conversations[session_id] = history[-max_messages:]


def format_history_for_prompt(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return "(no previous messages)"
    lines = []
    for msg in history:
        speaker = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def clear_history(session_id: str):
    _conversations.pop(session_id, None)