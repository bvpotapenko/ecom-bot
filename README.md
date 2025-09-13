# Shoply Support Bot

A conversational support bot built with **LangGraph** + **Gemini (Google Generative AI)**.
It uses structured prompts, style guidelines, and conversation memory to answer customer support questions for **Shoply**.

---

## Features

* **Conversational Support Bot** – built with LangGraph and LangChain, using Gemini models.
* **Structured Outputs** – responses formatted as JSON with `answer` and `actions` fields.
* **Order Commands** – `/order <order_number>` to retrieve order status.
* **Conversation Memory** – stateful threads with the ability to clear history (`clr`).
* **Configurable Style** – persona, tone, and fallback logic from `data/style_guide.yaml` and `data/prompts.yaml`.
* **Evaluation Script** – `tools/style_eval.py` tests model answers against the style guide.

---

## Project Structure

```
src/
  bot/
    app.py           # Main entry point for the bot (CLI)
    config.py        # Configuration values and messages
    order_utils.py   # Order loading and prompt building
data/
  prompts.yaml       # Prompt templates and greetings
  style_guide.yaml   # Brand persona & tone definitions
  few_shots.jsonl    # Few-shot training examples
  faq.json           # FAQ data used by the bot
  eval_prompts.txt   # prompts for the eval tool
  orders.json        # A sample dataset with orders info
lang/
  messages.json      # System replies are saved here
tools/
  style_eval.py      # Style evaluation script
reports/             # Evaluation reports will be saved here
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Install dependencies**:

   Using Poetry (recommended):

   ```bash
   poetry install
   ```

   Or with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   Copy `.env.example` to `.env` (if provided) and set:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   MODEL_NAME=gemini-2.5-flash   # optional, defaults to gemini-2.5-flash
   ```

---

## Running the Bot

You can run the bot interactively in the terminal:

```bash
poetry run python -m src.bot.app
```

(or, without poetry):

```bash
python -m src.bot.app
```

You’ll see:

```
Welcome, how can I help you today?
Type:
  /order <order_number> to get order status
  'exit', 'quit', 'bye', 'q', 'stop' or 'end' to quit
  'clr' to change topic
Or ask any question about order processing
```

**Special Commands**:

* `/order <order_number>` – fetch order info
* `clr` – clear conversation thread
* `exit`/`quit`/`bye` – quit the program

---

## Evaluating Style Compliance

The repository includes a style evaluation tool:

```bash
poetry run python -m tools.style_eval
```

(or):

```bash
python -m tools.style_eval
```

This will:

* Load prompts from `data/eval_prompts.txt`
* Generate answers using the bot
* Score them using:

  * **Rule-based checks** (length, emojis, exclamation marks)
  * **LLM-based checks** (using the style guide)
* Save results to `reports/style_eval.json`
* Print the mean score to the console

---

## Configuration Files

* **`data/prompts.yaml`** – defines system prompt, greetings, and prompt versions.
* **`data/style_guide.yaml`** – brand persona, tone, and prohibited/required items.
* **`data/faq.json`** – FAQ database.
* **`data/few_shots.jsonl`** – few-shot examples.

---

## Logging

* All sessions and errors are logged as `.jsonl` files in `./logs/`.
* Log level can be adjusted with `--log-level`:

```bash
python -m src.bot.app --log-level DEBUG
```

---

## Requirements

* Python 3.10+
* `langgraph`, `langchain-core`, `langchain-google-genai`, `pydantic`, `pyyaml`, `python-dotenv`
* A valid Google Generative AI API key.

---

## License

MIT License (or your preferred license).

---
