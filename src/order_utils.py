import json
import config


def load_orders():
    with open("data/orders.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_order_prompt(order_id: str, order: dict) -> str:
    """Generate LLM prompt for order status from JSON using template in messages.json."""
    order_json = json.dumps(order, ensure_ascii=False, indent=2)
    return config.MESSAGES["order_prompt_template"].format(
        order_id=order_id, order_json=order_json
    )
