"""Environment-driven configuration for the agent service."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# LLM_PROVIDER selects which OpenAI-compatible chat completions endpoint to use.
# OpenRouter and Google's Gemini API (via its OpenAI-compat endpoint) both accept
# the same request/response shape, so agent.py doesn't need to know which one
# it's talking to — just LLM_API_KEY / LLM_MODEL / LLM_BASE_URL below.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

if LLM_PROVIDER == "gemini":
    LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
else:
    LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    LLM_BASE_URL = "https://openrouter.ai/api/v1"

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://127.0.0.1:8000")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://127.0.0.1:8001")

SESSION_TTL_HOURS = float(os.getenv("SESSION_TTL_HOURS", "6"))
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "5"))

SESSIONS_DB_PATH = BASE_DIR / "sessions.db"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DB_PATH = LOGS_DIR / "turns.db"

# Voice (Phase 3) — both run fully locally/offline once their models are cached.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
VOICE_MODELS_DIR = BASE_DIR / "voice_models"
VOICE_OUTPUT_DIR = BASE_DIR / "voice_output"

# Plausibility bounds used by guardrails.validate_price when a city has too few
# listings in the local snapshot to derive a sensible per-city range.
GLOBAL_MIN_RENT_MAD = 500
GLOBAL_MAX_RENT_MAD = 100_000
