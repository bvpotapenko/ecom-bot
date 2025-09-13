"""Configuration for a support bot."""

from pathlib import Path
import json

REQUEST_TIMEOUT = 15  # Seconds for API request timeout
TEMPERATURE = 0.0  # LLM temperature for deterministic responses
MAX_OUTPUT_TOKENS = 1500  # Max tokens per LLM response
THINKING_BUDGET = 0  # Max tokens for LLM internal "thinking"


# Project root = two levels up from /bot
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Path to /lang/messages.json, relative to the project root
_messages_path = PROJECT_ROOT / "lang" / "messages.json"

# Message templates
with open(_messages_path, "r", encoding="utf-8") as f:
    MESSAGES = json.load(f)
