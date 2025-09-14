import json
import yaml
from typing import Tuple
from . import config


def load_orders():
    with open("data/orders.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_order_details(order_id: str, order: dict) -> str:
    """
    Get order status and other knowk details from JSON.
    """
    order_json = json.dumps(order, ensure_ascii=False, indent=2)

    # ---- Safely embed JSON Order Details ----
    # Escape braces so LangChain does NOT interpret as variables
    order_details_json_escaped = order_json.replace("{", "{{").replace("}", "}}")

    return order_details_json_escaped
