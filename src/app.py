"""
===========================================================
Shoply Support Bot – LangGraph + Gemini
===========================================================
"""

# ===========================
# Imports
# ===========================
import argparse
import json
import logging
import os
import sys
import getpass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict

# LangGraph / LangChain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel, Field

# Local modules
import config
from order_utils import load_orders, get_order_prompt

load_dotenv()


# ===========================================================
# Utilities: Logging, CLI Args, Env Keys
# ===========================================================
def parse_log_level(value: str) -> int:
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
    parser = argparse.ArgumentParser(description="Shoply support bot")
    parser.add_argument("--log-level", default="INFO", type=parse_log_level)
    parser.add_argument("--scenario", type=str)
    return parser.parse_args()


def setup_logging(log_level: int) -> None:
    """
    JSONL logging with rotating files.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    error_log_file = "./logs/error.jsonl"

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

    session_handler = RotatingFileHandler(log_file, maxBytes=1 * 1024 * 1024)
    session_handler.setFormatter(JsonlFormatter())
    session_handler.setLevel(logging.INFO)
    logger.addHandler(session_handler)

    error_handler = RotatingFileHandler(error_log_file, maxBytes=1 * 1024 * 1024)
    error_handler.setFormatter(JsonlFormatter())
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    if log_level <= logging.DEBUG:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)


def setup_gemini() -> None:
    """
    Prompt for Google API key if not set.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set.")


def load_system_prompt() -> str:
    """
    Loads brand FAQ into system prompt.
    """
    with open("data/faq.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    faq_str = json.dumps(faq_data, ensure_ascii=False)

    # ---- Safely embed JSON FAQ in system prompt ----
    # Escape braces so LangChain does NOT interpret as variables
    faq_json_escaped = faq_str.replace("{", "{{").replace("}", "}}")

    brand_name = os.environ.get("BRAND_NAME")
    if not brand_name:
        raise RuntimeError("Environment variable 'BRAND_NAME' is not set.")
    return config.SYSTEM_INSTRUCTION_TEMPLATE.format(
        brand=brand_name, faq=faq_json_escaped
    )


def get_new_thread_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ===========================================================
# LangGraph Setup – Nodes and Edges
# ===========================================================
def setup_langgraph(system_prompt: str) -> StateGraph:
    """
    Build a LangGraph workflow:
    START → LLM Node (system+messages) → END
    With MemorySaver as checkpoint.
    """
    # ---- LLM initialization ----
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        thinking_budget=config.THINKING_BUDGET,
        timeout=config.REQUEST_TIMEOUT,
    )

    # ---- Prompt Template ----
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )

    # ---- Node function ----
    def llm_node(state: MessagesState) -> Dict[str, List]:
        """
        Combines messages + system prompt → LLM.
        """

        chain = prompt_template | llm
        bot_resp = chain.invoke({"messages": state["messages"]})
        print(
            f"\n\n=========DEBUG:==========\n\nllm_node function: \n\n{bot_resp=}\n\n"
        )
        # Wrap back into MessagesState
        return {"messages": bot_resp}

    # ---- Graph building ----
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("llm_node", llm_node)
    workflow.add_edge(START, "llm_node")
    workflow.add_edge("llm_node", END)

    # ---- Memory Checkpointer (conversation history) ----
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


# ===========================================================
# 4. Core message processing
# ===========================================================
def process_message(
    graph: StateGraph, graph_config: dict, input_text: str, logger: logging.Logger
) -> Tuple[str, int]:
    """
    Run a single user message through the compiled LangGraph and extract
    a textual bot reply and token usage.

    Parameters
    ----------
    graph : StateGraph
        Compiled LangGraph workflow (StateGraph.compile(...)).
    graph_config : dict
        Runtime config passed to graph.invoke(), e.g. {"configurable": {"thread_id": "..."}}
    input_text : str
        Raw user message or prompt text (e.g., order prompt).
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    Tuple[str, int]
        (bot_reply, total_tokens). If an error occurs, bot_reply will be
        a user-friendly error message and total_tokens will be 0.
    """
    total_tokens = 0
    bot_reply = config.MESSAGES.get("unexpected_error", "Sorry, something went wrong.")

    try:
        with get_usage_metadata_callback() as usage_callback:
            # 1) Build a MessagesState-compatible input.
            # MessagesState expects {"messages": [ChatMessage, ...]}
            input_state = {"messages": [HumanMessage(content=input_text)]}

            # 2) Run the workflow synchronously. The graph returns a dict "state".
            state = graph.invoke(input_state, graph_config, callbacks=[usage_callback])
            print(
                f"\n\n=========DEBUG:==========\n\nprocess message function: \n\n{state=}\n\n"
            )
            # 3) Extract usage metadata safely (guard against empty metadata).
            usage_data = getattr(usage_callback, "usage_metadata", {}) or {}
            model_stats = {}
            if usage_data:
                # usage_data is typically a dict of {callback_id: metadata}
                first_key = next(iter(usage_data))
                model_stats = usage_data[first_key] or {}
                total_tokens = int(model_stats.get("total_tokens", 0))

            # 4) Extract "messages"
            # The node added "messages" to state
            bot_reply = state["messages"][-1].content
            logger.info(
                {"prompt": input_text, "bot_reply": bot_reply, "usage": model_stats}
            )
    except Exception as e:
        bot_reply = config.MESSAGES["unexpected_error"]
        logger.exception("Error during process_message")
        logger.error({"error": f"Unexpected error: {str(e)}"})

    return bot_reply, total_tokens


# ===========================================================
# 5. Special Commands
# ===========================================================
def handle_special_command(
    user_input: str, graph: StateGraph, graph_config: dict, logger: logging.Logger
) -> Optional[Tuple[str, int, bool, Optional[dict]]]:
    """
    Handle short CLI-style special commands.

    Parameters
    ----------
    user_input : str
        Raw text the user typed (e.g., "/order 123", "clr", "exit").
    graph : StateGraph
        Compiled LangGraph workflow object. Used only if the command needs to call the model
        (e.g. /order).
    graph_config : dict
        Graph runtime configuration passed to graph.invoke(). Expected structure:
            {"configurable": {"thread_id": "<thread_id>"}}
        Used for isolatoin of different conversation threads in the checkpointer.
    logger : logging.Logger
        Python logger for structured logging.

    Returns
    -------
    Optional[Tuple[str, int, bool, Optional[dict]]]
        - If `None`: the input was NOT a special command and normal processing should continue.
        - Otherwise returns a 4-tuple:
            (bot_reply: str,
            tokens: int,           # tokens consumed by any model call; 0 if none
            exit_flag: bool,       # if True -> caller should terminate loop
            new_config: Optional[dict])  # if provided, caller should replace current graph_config
    """
    if user_input.startswith("/order"):
        parts = user_input.split()
        if len(parts) != 2:
            return config.MESSAGES["order_usage"], 0, False, None
        order_id = parts[1]
        orders = (
            load_orders()
        )  # returns dict[str, Any] — mapping order_id -> order dict
        order = orders.get(order_id)
        if order:
            order_prompt = get_order_prompt(
                order_id, order
            )  # returns str (prompt text)
            bot_reply, tokens = process_message(
                graph, graph_config, order_prompt, logger
            )
            return bot_reply, tokens, False, None
        else:
            return (
                config.MESSAGES["order_not_found"].format(order_id=order_id),
                0,
                False,
                None,
            )

    if user_input.lower() in {"exit", "quit", "bye", "q", "stop", "end"}:
        return config.MESSAGES["goodbye"], 0, True, None

    if user_input.lower() == "clr":
        new_thread_id = get_new_thread_id()
        new_config = {"configurable": {"thread_id": new_thread_id}}
        return config.MESSAGES["clear"], 0, False, new_config

    return None


# ===========================================================
# 6. CLI Loop
# ===========================================================
def chat_loop(graph: StateGraph, initial_config: dict, logger: logging.Logger) -> None:
    """
    Interactive console loop.
    """
    current_config = initial_config
    total_tokens_count = 0

    logger.info({"greetings": config.MESSAGES["greetings"]})
    print(config.MESSAGES["greetings"])

    while True:
        try:
            user_input = input("You: ").strip()
            logger.info({"user_input": user_input})
        except (KeyboardInterrupt, EOFError):
            print("\n" + config.MESSAGES["goodbye"])
            logger.info({"total_tokens_count": total_tokens_count})
            break

        if not user_input:
            continue

        special_result = handle_special_command(
            user_input, graph, current_config, logger
        )
        if special_result:
            bot_reply, tokens, exit_flag, new_config = special_result
            total_tokens_count += tokens
            print("\nBot: " + bot_reply)
            if exit_flag:
                logger.info({"total_tokens_count": total_tokens_count})
                break
            if new_config:
                current_config = new_config
            continue

        bot_reply, tokens = process_message(graph, current_config, user_input, logger)
        total_tokens_count += tokens
        print("Bot: " + bot_reply)


# ===========================================================
# 7. Entry Point
# ===========================================================
def main() -> int:
    args = setup_argparse()
    setup_logging(args.log_level)
    setup_gemini()
    system_prompt = load_system_prompt()
    graph = setup_langgraph(system_prompt)
    initial_thread_id = get_new_thread_id()
    initial_config = {"configurable": {"thread_id": initial_thread_id}}
    chat_loop(graph, initial_config, logging.getLogger())
    return 0


if __name__ == "__main__":
    sys.exit(main())
