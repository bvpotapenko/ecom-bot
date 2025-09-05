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
from langchain.schema import SystemMessage
from langchain_core.callbacks import get_usage_metadata_callback
import config
from order_utils import load_orders, get_order_prompt

load_dotenv()


def parse_log_level(value: str) -> int:
    """Parse log level string to logging level constant."""
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
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(description="Shoply support bot")
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=parse_log_level,
        help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG",
    )
    return parser.parse_args()


def setup_logging(log_level: int) -> str:
    """Configure logging to JSONL format in logs/session_<timestamp>.jsonl and errors to error.jsonl."""
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
    """Prompt for Google API key if not set in environment."""
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")


def load_system_prompt() -> str:
    """Load FAQ and format system prompt."""
    with open("data/faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    faq_str = json.dumps(faq_data, ensure_ascii=False)
    return config.SYSTEM_INSTRUCTION_TEMPLATE.format(
        brand=os.environ["BRAND_NAME"], faq=faq_str
    )


def chat_loop(conversation: ConversationChain, logger: logging.Logger) -> None:
    """Run the main chat loop, handling user input and bot responses."""
    usage_callback = get_usage_metadata_callback()
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

        user_input = user_input.strip()
        if not user_input:
            continue

        # process /order command
        if user_input.startswith("/order"):
            try:
                parts = user_input.split()
                if len(parts) != 2:
                    bot_reply = config.MESSAGES["order_usage"]
                else:
                    order_id = parts[1]
                    orders = load_orders()
                    order = orders.get(order_id)
                    if order:
                        order_prompt = get_order_prompt(order_id, order)
                        with get_usage_metadata_callback() as usage_callback:
                            bot_reply = conversation.predict(
                                input=order_prompt, callbacks=[usage_callback]
                            )
                            usage_data = usage_callback.usage_metadata
                            _, model_stats = next(iter(usage_data.items()))

                            log_entry = {
                                "prompt": order_prompt,
                                "bot_reply": bot_reply,
                                "usage": model_stats,
                            }
                            total_tokens_count += log_entry["usage"]["total_tokens"]
                    else:
                        bot_reply = config.MESSAGES["order_not_found"].format(
                            order_id=order_id
                        )
            except Exception as e:
                bot_reply = config.MESSAGES["error_order"]
                logger.error({"error": f"Order error: {e}"})
            logger.info({"bot_reply": bot_reply})
            print("\nBot: " + bot_reply)
            continue

        # handle exit commands
        if user_input.lower() in {"exit", "quit", "bye", "q", "stop", "end"}:
            bot_reply = config.MESSAGES["goodbye"]
            logger.info({"bot_reply": bot_reply})
            print("\nBot: " + bot_reply)
            logger.info(
                {
                    "bot_reply": "Total tokens used during the session",
                    "total_tokens": total_tokens_count,
                }
            )
            break

        # handle clear command
        if user_input.lower() == "clr":
            conversation.memory.clear()
            system_prompt = load_system_prompt()
            conversation.memory.chat_memory.add_message(
                SystemMessage(content=system_prompt)
            )
            bot_reply = config.MESSAGES["clear"]
            logger.info({"bot_reply": bot_reply})
            print("\nBot: " + bot_reply)
            continue

        # get bot reply from LLM
        try:
            with get_usage_metadata_callback() as usage_callback:
                bot_reply = conversation.predict(
                    input=user_input, callbacks=[usage_callback]
                )
                usage_data = usage_callback.usage_metadata
                _, model_stats = next(iter(usage_data.items()))

                log_entry = {
                    "bot_reply": bot_reply,
                    "usage": model_stats,
                }
                total_tokens_count += log_entry["usage"]["total_tokens"]
        except Exception as e:
            bot_reply = config.MESSAGES["unexpected_error"]
            log_entry = {"bot_reply": bot_reply, "error": str(e)}
            logger.error({"error": f"Unexpected error: {str(e)}"})
        logger.info(log_entry)
        print(f"Bot: {bot_reply}")


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
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    conversation.memory.chat_memory.add_message(SystemMessage(content=system_prompt))

    chat_loop(conversation, logging.getLogger())
    return 0


if __name__ == "__main__":
    sys.exit(main())
