# Shoply Support Bot

A console-based customer support bot for the Shoply e-commerce platform, powered by LangChain and Google Gemini. It handles order status queries and FAQ-based questions, logging interactions in JSONL format.

## Features
- **Order Status Queries**: Retrieve and summarize order status using `/order <order_id>` command.
- **FAQ Support**: Answers questions based on predefined FAQs in `data/faq.json`.
- **Conversation Memory**: Maintains context for coherent interactions, with a `clr` command to reset.
- **Logging**: Logs interactions to `logs/session_<timestamp>.jsonl` and errors to `logs/error.jsonl`.
- **Configurable**: Adjustable via environment variables and `config.py`.

## Requirements
- Python 3.8+
- Google Gemini API key
- Poetry for dependency management
- Dependencies: `langchain`, `langchain-google-genai`, `python-dotenv`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd shoply-support-bot
   ```
2. Install Poetry (if not already installed):
   ```bash
   pip install poetry
   ```
3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
4. Set up environment variables:
   ```bash
   export GOOGLE_API_KEY=<your_api_key>
   export BRAND_NAME=Shoply
   export MODEL_NAME=gemini-2.5-flash
   ```
   Alternatively, create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=<your_api_key>
   BRAND_NAME=Shoply
   MODEL_NAME=gemini-2.5-flash
   ```

## Usage
1. Activate the Poetry virtual environment:
   ```bash
   poetry shell
   ```
2. Run the bot:
   ```bash
   python src/app.py --log-level INFO
   ```
3. Interact with the bot:
   - Type `/order <order_id>` to check order status (e.g., `/order 12345`).
   - Ask questions based on FAQs.
   - Use `clr` to clear conversation history.
   - Use `exit`, `quit`, `bye`, `q`, `stop`, or `end` to quit.

## Project Structure
- `src/app.py`: Main application script.
- `src/config.py`: Configuration settings and message templates.
- `src/order_utils.py`: Utilities for loading and processing orders.
- `data/faq.json`: FAQ data for bot responses.
- `data/orders.json`: Sample order data.
- `lang/messages.json`: Message templates for bot interactions.
- `logs/`: Directory for session and error logs.
- `poetry.lock`: Lock file for reproducible dependency versions.

## Logging
- Session logs: `logs/session_<timestamp>.jsonl` (INFO and above).
- Error logs: `logs/error.jsonl` (ERROR and CRITICAL).
- Log level can be set via `--log-level` (CRITICAL, ERROR, WARNING, INFO, DEBUG).

## Example Interaction
```plaintext
At your service. Type:
 /order <order_number> to get order status
 'exit', 'quit', 'bye', 'q', 'stop' or 'end' to quit
 'clr' to change topic
 Or ask any question about order processing.
You: /order 12345
Bot: Your order 12345 is in transit via ShoplyExpress, expected to arrive in 2 days.
You: exit
Bot: Goodbye! Have a nice day!
```

## License
MIT License