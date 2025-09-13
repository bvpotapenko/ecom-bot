"""Configuration for a support bot."""

import os
import json

REQUEST_TIMEOUT = 15  # Seconds for API request timeout
TEMPERATURE = 0.0  # LLM temperature for deterministic responses
MAX_OUTPUT_TOKENS = 1500  # Max tokens per LLM response
THINKING_BUDGET = 0  # Max tokens for LLM internal "thinking"


# Project root = one level up from /src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to /lang/messages.json
_messages_path = os.path.join(PROJECT_ROOT, "lang", "messages.json")

# Message templates
with open(_messages_path, "r", encoding="utf-8") as f:
    MESSAGES = json.load(f)
