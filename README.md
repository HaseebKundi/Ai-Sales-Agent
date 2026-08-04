# AI Customer Support Agent

A production-shaped AI support agent for small e-commerce/SaaS businesses. It answers order-status and FAQ questions automatically, and **escalates to a human** the moment it isn't confident — instead of guessing.

Built with **LangGraph + FastAPI + SQLite/SQLAlchemy + Chroma (RAG) + Groq**.

---

## Features

- **Intent classification** — every message is classified as `order_status`, `faq`, `complaint`, or `other` by an LLM node.
- **Order lookup (mock order system)** — a SQLAlchemy/SQLite `orders` table simulates a real order database. The agent extracts an order ID (e.g. `ORD-1001`) from the message and looks it up.
- **FAQ RAG** — store policies (returns, shipping, payments, cancellations) live in `data/faq.json`, embedded and indexed in a persisted **Chroma** vector store. Questions are answered only from retrieved context — no hallucinated policy.
- **Confidence-gated escalation (the important part)** — after every lookup or retrieval, a `decide_confidence` node checks a similarity/confidence score against a threshold (`CONFIDENCE_THRESHOLD`, default `0.55`). Below it, the agent **does not answer** — it hands off to a human instead. Complaints and unclassifiable messages skip straight to escalation.
- **Automatic ticket creation** — every escalation writes a row to a `support_tickets` table with the reason, intent, and original message, so a human has a ready-made inbox (`GET /tickets`).
- **FastAPI backend** — clean REST API (`/chat`, `/tickets`, `/health`) with interactive docs at `/docs`.
- **Chat widget** — a small vanilla HTML/JS widget (`static/index.html`) served at `/`, showing intent, confidence, and escalation status per reply — good for demos.
- **Free-tier friendly** — Groq for fast/free LLM inference, FastEmbed (ONNX, no torch) for local embeddings, SQLite for storage. No paid services required to run this end to end.

## How it works — the LangGraph flow

```
                     ┌────────────────┐
                     │ classify_intent │
                     └───────┬────────┘
             ┌───────────────┼───────────────────┐
             ▼                ▼                    ▼
     order_status           faq              complaint / other
             │                │                    │
     ┌───────▼──────┐  ┌──────▼──────┐              │
     │ lookup_order │  │  check_faq  │              │
     └───────┬──────┘  └──────┬──────┘              │
             └────────┬───────┘                     │
                       ▼                             │
             ┌───────────────────┐                   │
             │ decide_confidence │                   │
             └─────────┬─────────┘                   │
          confidence ≥ threshold │ < threshold        │
                       ▼         ▼                    ▼
             ┌────────────────┐ ┌─────────────────────────┐
             │ answer_directly│ │        escalate          │
             │  (LLM answer)  │ │ (create ticket, hand off)│
             └───────┬────────┘ └─────────────┬───────────┘
                     ▼                         ▼
                    END                       END
```

This is the part that matters for real deployments: **the agent is allowed to say "I don't know, let me get a human"** instead of confidently answering with wrong information. That's what separates a usable support bot from a happy-path demo.

## Project structure

```
ai-support-agent/
├── app/
│   ├── main.py         # FastAPI app: /chat, /tickets, /health
│   ├── agent.py         # LangGraph graph: nodes + routing + escalation logic
│   ├── rag.py            # Chroma vector store + FAQ similarity search
│   ├── models.py          # SQLAlchemy models: Order, SupportTicket
│   ├── database.py         # SQLAlchemy engine/session
│   ├── schemas.py           # Pydantic request/response models
│   ├── seed_data.py          # Seeds 4 sample orders (ORD-1001..1004)
│   └── config.py               # Env-based settings
├── data/faq.json         # FAQ knowledge base (source for RAG)
├── static/index.html      # Chat widget demo UI
├── requirements.txt
├── render.yaml              # One-click Render deployment config
├── .env.example
└── README.md
```

## Setup (local)

**Requirements:** Python 3.11+, a free [Groq API key](https://console.groq.com/keys).

```bash
git clone <your-repo-url>
cd ai-support-agent

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GROQ_API_KEY
```

Run it:

```bash
uvicorn app.main:app --reload
```

- Chat widget: **http://localhost:8000/**
- API docs (Swagger): **http://localhost:8000/docs**
- Escalated tickets: **http://localhost:8000/tickets**

The database is seeded automatically on first startup with 4 sample orders. The FAQ vector store is built automatically on first use (downloads a small embedding model on first run only, then caches it).

### Sample data to try

| Order ID  | Status      |
|-----------|-------------|
| ORD-1001  | shipped     |
| ORD-1002  | delivered   |
| ORD-1003  | processing  |
| ORD-1004  | cancelled   |

Try messages like:
- `"Where is my order ORD-1001?"` → answered directly (order found)
- `"What's your return policy?"` → answered directly (confident FAQ match)
- `"Where's my order ORD-9999?"` → escalated (order not found)
- `"This product arrived broken and I want a refund now"` → escalated (complaint)
- `"asdkjfh random gibberish"` → escalated (unclear intent)

## API

### `POST /chat`
```json
// request
{ "message": "Where is my order ORD-1001?" }

// response
{
  "response": "Your order ORD-1001 (Wireless Headphones) has shipped via FedEx...",
  "intent": "order_status",
  "escalated": false,
  "escalation_reason": null,
  "confidence": 0.95,
  "order_info": { "order_id": "ORD-1001", "status": "shipped", "...": "..." }
}
```

### `GET /tickets`
Returns every auto-created escalation ticket — the human agent's queue.

### `GET /health`
Basic liveness check.

## Deploying to Render

This repo includes `render.yaml` for one-click deployment via Render's Blueprint feature.

1. Push this project to a GitHub repo.
2. In Render, click **New → Blueprint**, and point it at your repo (Render will read `render.yaml` automatically).
3. Set the `GROQ_API_KEY` environment variable in the Render dashboard (it's marked `sync: false` so it isn't committed to git).
4. Deploy. Render will run `pip install -r requirements.txt` and start with:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Your live demo will be at `https://<your-service>.onrender.com/`.

**Note on Render's free tier:** the disk is ephemeral — SQLite data and the Chroma index reset on redeploys. That's fine for a demo; for a real client deployment, swap `DATABASE_URL` for a managed Postgres (e.g. Render's free Postgres) and mount a persistent disk for `CHROMA_DIR`, or use Render's paid persistent disks. The code doesn't need to change — just the connection string.

## Extending this for a real client

- Swap the mock `Order` table for a call to their real order/shipping API.
- Add authentication (API key or JWT) to `/chat`.
- Add a webhook in `escalate_to_human` to notify Slack/email when a ticket is created.
- Add conversation memory (LangGraph supports checkpointing) for multi-turn context.
- Swap FastEmbed for OpenAI/Groq embeddings if you want higher retrieval quality at the cost of an API call.

## Stack

| Layer            | Tool                                |
|-------------------|--------------------------------------|
| Agent orchestration | LangGraph                          |
| LLM inference      | Groq (`llama-3.3-70b-versatile`)    |
| API                | FastAPI                             |
| Database           | SQLite + SQLAlchemy                 |
| Vector store        | Chroma (persisted locally)         |
| Embeddings          | FastEmbed (`bge-small-en-v1.5`, local, free) |
| Deployment          | Render                             |
