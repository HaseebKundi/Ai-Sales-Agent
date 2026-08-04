import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./support.db")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    # Below this similarity/confidence score, the agent escalates to a human
    # instead of guessing.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))


settings = Settings()
