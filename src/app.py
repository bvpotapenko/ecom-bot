"""Console-based support bot for Shoply e-commerce using LangChain and Gemini."""

import argparse
import json
import logging
import os
import sys
import getpass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.callbacks import get_usage_metadata_callback
from typing import Optional, Tuple, List, Dict
import yaml
from pydantic import BaseModel, Field
import config
from order_utils import load_orders, get_order_prompt

load_dotenv()

# ===========================
# Bot Response Schema
# ===========================


class BotResponse(BaseModel):
    answer: str = Field(description="Краткий ответ")
    tone: str = Field(
        description="Контроль: совпадает ли тон (да/нет) + одна фраза почему"
    )
    actions: List[str] = Field(
        description="Список следующих шагов для клиента (0–3 пункта)"
    )


# ===========================
# Helpers
# ===========================


def parse_log_level(value: str) -> int:
    """
    Parse log level string to logging level constant.
    Raises argparse.ArgumentTypeError for invalid values.
    """
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    level = mapping.get(value.upper())
    if level is None:
        raise argparse.ArgumentTypeError(
            f"Invalid log level '{value}'. Choose from {', '.join(mapping)}."
        )
    return level


def setup_argparse() -> argparse.Namespace:
    """
    Setup command-line argument parsing.
    The commands are:
    --log-level: Set the logging level (default: INFO).
    --scenario: Path to a JSON file containing a list of user inputs for scenario mode.
    """
    parser = argparse.ArgumentParser(description="Shoply support bot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=parse_log_level,
        help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run in scenario mode using a JSON file of user inputs.",
    )
    return parser.parse_args()


def setup_logging(log_level: int) -> str:
    """
    Configure logging to JSONL format in logs/session_<timestamp>.jsonl and errors to error.jsonl.
    Returns the session log file path.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    # Create logs directory if it doesn't exist
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    error_log_file = "./logs/error.jsonl"

    # Custom JSONL formatter to handle both string and dict messages
    class JsonlFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
            }
            if isinstance(record.msg, dict):
                log_entry.update(record.msg)
            else:
                log_entry["message"] = record.msg
            return json.dumps(log_entry, ensure_ascii=False)

    # Session log handler (INFO and above)
    session_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    session_handler.setLevel(logging.INFO)
    session_handler.setFormatter(JsonlFormatter())
    logger.addHandler(session_handler)

    # Error log handler (ERROR and CRITICAL)
    error_handler = RotatingFileHandler(
        error_log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JsonlFormatter())
    logger.addHandler(error_handler)

    # Console logging for DEBUG
    if log_level <= logging.DEBUG:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(console_handler)

    return log_file


def setup_gemini() -> None:
    """
    Prompt for Google API key if not set in environment.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")


def load_system_prompt() -> str:
    """
    Load FAQ, format system prompt, few shots, and return the full system prompt.
    """
    with open("data/faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    faq_str = json.dumps(faq_data, ensure_ascii=False)

    # Load style guide
    with open("data/style_guide.yaml", "r", encoding="utf-8") as f:
        style_guide = yaml.safe_load(f)
    style_guide_str = yaml.dump(style_guide, allow_unicode=True)

    # Load few shots
    few_shots = []
    with open("data/few_shots.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            few_shots.append(
                f"User: {example['user']}\Assistant: {example['assistant']}"
            )
    few_shots_str = "\n\n".join(few_shots)

    return (
        config.SYSTEM_INSTRUCTION_TEMPLATE.format(
            brand=os.environ["BRAND_NAME"],
            faq=faq_str,
            style_guide=style_guide_str,
            few_shots=few_shots_str,
        )
        + "\nAlways respond in JSON format matching BotResponse schema."
    )


# ===========================
# Core Processing
# ===========================


def process_single_input(
    conversation: ConversationChain, user_input: str, logger: logging.Logger
) -> Tuple[str, Optional[Dict]]:
    """
    Process one user input and return (formatted_reply, usage_stats).
    usage_stats is a dict with model usage metadata (contains 'total_tokens' when available),
    or None for non-LLM flows.
    This centralizes order handling, clearing memory, exit text, and LLM calls.
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return "", None

    # Handle /order
    if user_input.startswith("/order"):
        try:
            parts = user_input.split()
            if len(parts) != 2:
                reply = config.MESSAGES["order_usage"]
                logger.info({"bot_reply": reply})
                return reply, None

            order_id = parts[1]
            orders = load_orders()
            order = orders.get(order_id)
            if order:
                order_prompt = get_order_prompt(order_id, order)
                with get_usage_metadata_callback() as usage_callback:
                    raw_reply = conversation.predict(
                        input=order_prompt, callbacks=[usage_callback]
                    )
                    parser = JsonOutputParser(pydentic_object=BotResponse)
                    structured_reply = parser.parse(raw_reply)
                    formatted_reply = f"{structured_reply['answer']}\n{'\n'.join(structured_reply['actions'])}"

                    usage_data = usage_callback.usage_metadata
                    model_stats = None
                    if usage_data:
                        _, model_stats = next(iter(usage_data.items()))
                    log_entry = {"prompt": order_prompt, "bot_reply": formatted_reply}
                    if model_stats:
                        log_entry["usage"] = model_stats
                    logger.info(log_entry)
                    return formatted_reply, model_stats
            else:
                reply = config.MESSAGES["order_not_found"].format(order_id=order_id)
                logger.info({"bot_reply": reply})
                return reply, None
        except Exception as e:
            logger.error({"error": f"Order error: {e}"})
            return config.MESSAGES["error_order"], None

    # Special text commands: exit/clear
    if user_input.lower() in {"exit", "quit", "bye", "q", "stop", "end"}:
        return config.MESSAGES["goodbye"], None

    if user_input.lower() == "clr":
        conversation.memory.clear()
        system_prompt = load_system_prompt()
        conversation.memory.chat_memory.add_message(
            SystemMessage(content=system_prompt)
        )
        reply = config.MESSAGES["clear"]
        logger.info({"bot_reply": reply})
        return reply, None

    # Generic LLM input
    try:
        with get_usage_metadata_callback() as usage_callback:
            raw_reply = conversation.predict(
                input=user_input, callbacks=[usage_callback]
            )
            parser = JsonOutputParser(pydentic_object=BotResponse)
            structured_reply = parser.parse(raw_reply)
            # Format for display
            formatted = (
                f"{structured_reply.answer}\nTone check: {structured_reply.tone}\nActions:\n"
                + "\n".join([f"- {a}" for a in structured_reply.actions])
            )
            # Log
            log_entry = {"bot_reply": formatted, "structured": structured_reply.dict()}
            usage_data = usage_callback.usage_metadata
            model_stats = None
            if usage_data:
                _, model_stats = next(iter(usage_data.items()))
            log_entry = {"bot_reply": bot_reply}
            if model_stats:
                log_entry["usage"] = model_stats
            logger.info(log_entry)
            return formatted, model_stats
    except Exception as e:
        logger.error({"error": f"Unexpected error: {str(e)}"})
        return config.MESSAGES["unexpected_error"], None


def run_chat_scenario(
    conversation: ConversationChain,
    scenario: List[str],
    logger: logging.Logger,
    reset_memory: bool = True,
    token_usage: Dict = None,
) -> Tuple[List[str], int]:
    """
    A suplimentary function for testing.
    Run a multi-turn chat scenario for style evaluation.
    - If reset_memory is True, the conversation's memory will be cleared and system prompt re-inserted.
    - Returns (replies, total_tokens) where total_tokens is the sum of total_tokens for LLM calls in the scenario.
    """

    total_tokens = 0
    replies: List[str] = []

    if reset_memory:
        conversation.memory.clear()
        system_prompt = load_system_prompt()
        conversation.memory.chat_memory.add_message(
            SystemMessage(content=system_prompt)
        )

    for turn in scenario:
        reply, usage = process_single_input(conversation, turn, logger)
        replies.append(reply)
        if usage and "total_tokens" in usage:
            total_tokens += usage["total_tokens"]
            if token_usage:
                token_usage["input_tokens"] += usage.get("input_tokens", 0)
                token_usage["output_tokens"] += usage.get("output_tokens", 0)

    logger.info(
        {
            "scenario_len": len(scenario),
            "total_tokens_for_scenario": total_tokens,
            "last_reply": replies[-1] if replies else None,
        }
    )
    return replies, total_tokens


# -----------------------------
# CLI loop
# -----------------------------


def chat_loop(
    conversation: ConversationChain, logger: logging.Logger, token_usage: Dict
) -> None:
    """
    Interactive CLI loop. Accumulates total tokens used during the session and logs them on exit.
    """
    total_tokens_count = 0

    # Log and print greetings
    logger.info({"greetings": config.MESSAGES["greetings"]})
    print(config.MESSAGES["greetings"])

    while True:
        try:
            user_input = input("You: ")
            logger.info({"user_input": user_input})
        except (KeyboardInterrupt, EOFError):
            bot_reply = config.MESSAGES["goodbye"]
            logger.info({"bot_reply": bot_reply})
            logger.info(
                {
                    "bot_reply": "Total tokens used during the session",
                    "total_tokens": total_tokens_count,
                }
            )
            print("\n" + bot_reply)
            break

        user_input = (user_input or "").strip()
        if not user_input:
            continue

        reply, usage = process_single_input(conversation, user_input, logger)

        # accumulate tokens if present
        if usage and isinstance(usage, dict):
            token_usage["input_tokens"] += usage.get("input_tokens", 0)
            token_usage["output_tokens"] += usage.get("output_tokens", 0)
            log_entry = {"bot_reply": reply, "usage": usage}
        else:
            log_entry = {"bot_reply": reply}

        logger.info(log_entry)
        print(f"\nBot: {reply}")

        # If the user asked to exit, break loop (process_single_input already returned goodbye)
        if user_input.lower() in {"exit", "quit", "bye", "q", "stop", "end"}:
            logger.info(
                {
                    "bot_reply": "Total tokens used during the session",
                    "total_tokens": total_tokens_count,
                }
            )
            break


# ===========================
# Entry Point
# ===========================


def main() -> int:
    """Main function to initialize and run the Shoply support bot."""
    args = setup_argparse()

    log_file = setup_logging(args.log_level)

    setup_gemini()

    system_prompt = load_system_prompt()

    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        thinking_budget=config.THINKING_BUDGET,
        timeout=config.REQUEST_TIMEOUT,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

    token_usage = {"input_tokens": 0, "output_tokens": 0}

    conversation.memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    token_usage = {"input_tokens": 0, "output_tokens": 0}

    logger = logging.getLogger()

    if args.scenario:
        with open(args.scenario, "r", encoding="utf-8") as f:
            scenario = json.load(f)

        replies = run_chat_scenario(conversation, scenario, logger, token_usage)

        print("\n--- Scenario Run Completed ---")
        for i, (u, r) in enumerate(zip(scenario, replies)):
            print(f"Turn {i+1} | You: {u}")
            print(f"       | Bot: {r}\n")

    else:
        chat_loop(conversation, logger, token_usage)

    # Final usage summary
    logger.info(
        {
            "total_tokens": token_usage["input_tokens"] + token_usage["output_tokens"],
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
