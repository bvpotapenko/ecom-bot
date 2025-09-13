"""
===========================================================
A module to evaluate Shoply Support Bot following style guidelines
===========================================================

Run the module as a script to evaluate prompts in 'data/eval_prompts.txt'.

Steps performed:
    1. Set up logging.
    2. Read prompts from 'data/eval_prompts.txt'.
    3. Call eval_batch() with the loaded prompts.
    4. Print the mean score and report location to stdout.
"""

import json
import pathlib
import re
from datetime import datetime
import os
import statistics
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import yaml

from src.bot.app import (
    load_system_prompt,
    setup_langgraph,
    process_message,
    handle_special_command,
)
import src.bot.config

# Setup paths and environment
BASE = pathlib.Path(__file__).parent.parent
load_dotenv(BASE / ".env", override=True)
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)


# Simple rule-based checks
def rule_checks(text: str) -> int:
    """
    Evaluate a piece of text against simple, rule-based style checks.

    Main logic and steps:
        1. Start with a base score of 100.
        2. Deduct 20 points if any emojis (Unicode range U+1F300–U+1FAFF) are present.
        3. Deduct 10 points if three or more consecutive exclamation marks ("!!!") appear.
        4. Deduct 10 points if the text length exceeds 600 characters.
        5. Return the resulting score, clamped at a minimum of 0.

    Args:
        text (str): The text to be evaluated.

    Returns:
        int: A rule-based style score from 0 to 100, where higher values indicate better compliance.
    """
    score = 100
    if re.search(r"[\U0001F300-\U0001FAFF]", text):  # No emojis
        score -= 20
    if "!!!" in text:  # No excessive exclamation
        score -= 10
    if len(text) > 600:  # Length limit
        score -= 10
    return max(score, 0)


# LLM-based evaluation
class Grade(BaseModel):
    """
    Data model representing the output of the LLM-based evaluation.
    """

    score: int = Field(
        ...,
        description="A numeric grade between 0 and 100 assigned by the LLM.",
        ge=0,
        le=100,
    )
    notes: str = Field(
        description="Short textual explanation or justification for the score"
    )


def setup_llm_grade_prompt() -> ChatPromptTemplate:
    """
    Build and return a LangChain ChatPromptTemplate for LLM grading.

    Main logic and steps:
        1. Load brand style and tone information from 'data/style_guide.yaml'.
        2. Extract persona, prohibited words, and required phrases.
        3. Construct a multi-message prompt template guiding the LLM to:
            - act as a strict reviewer,
            - score the assistant's response from 0 to 100,
            - provide short explanatory notes.

    Returns:
        ChatPromptTemplate: A prompt template that can be invoked with
        {'answer': <text>} to get a score and notes from the LLM.
    """
    with open("data/style_guide.yaml", "r", encoding="utf-8") as f:
        style_guide = yaml.safe_load(f)
    brand = style_guide["brand"]
    tone = style_guide["tone"]
    persona = tone["persona"]
    avoid = ", ".join(tone["avoid"])
    must_include = ", ".join(tone["must_include"])

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a strict reviewer ensuring responses align with the {brand} brand voice.",
            ),
            (
                "system",
                f"Tone: {persona}. Avoid: {avoid}. " f"Must include: {must_include}.",
            ),
            (
                "human",
                "Assistant response:\n{answer}\n\nProvide an integer score (0-100) and brief notes explaining the score.",
            ),
        ]
    )


def llm_grade(text: str, llm) -> Grade:
    """
    Use the LLM to grade a piece of text according to the style guide.

    Main logic and steps:
        1. Obtain the grading prompt from setup_llm_grade_prompt().
        2. Wrap the LLM with a structured output parser targeting the Grade model.
        3. Invoke the prompt with the text as input to produce a Grade object.

    Args:
        text (str): The text to be graded.
        llm: A LangChain-compatible LLM instance supporting 'with_structured_output'.

    Returns:
        Grade: The score (0–100) and notes produced by the LLM.
    """
    parser = llm.with_structured_output(Grade)
    prompt = setup_llm_grade_prompt()
    return (prompt | parser).invoke({"answer": text})


def parse_reply(reply_str: str) -> tuple[Optional[str], list[str]]:
    """
    Parse a bot reply string into an answer and a list of actions.

    Main logic and steps:
        1. Split the reply into lines.
        2. Detect and extract the value of 'answer:' (expects a quoted string).
        3. Collect subsequent lines under 'actions:' as a list of actions.
        4. Return both the extracted answer and actions.

    Args:
        reply_str (str): Raw string reply from the bot.

    Returns:
        tuple[Optional[str], list[str]]:
            - The extracted answer text (or None if parsing failed).
            - A list of action strings following 'actions:'.
    """
    lines = reply_str.splitlines()
    answer = None
    actions = []
    current = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("answer:"):
            try:
                answer = stripped.split('"', 1)[1].rsplit('"', 1)[0]
            except IndexError:
                pass
        elif stripped.startswith("actions:"):
            current = "actions"
        elif current == "actions" and stripped:
            actions.append(stripped)
    return answer, actions


def eval_batch(prompts: List[str], logger) -> dict:
    """
    Evaluate a batch of prompts using both rule-based and LLM-based grading.

    Main logic and steps:
        1. Initialize the LangGraph pipeline and system prompt.
        2. Configure a unique thread ID for this evaluation run.
        3. Create an LLM instance for grading.
        4. For each prompt:
            a. Get the bot’s reply via process_message().
            b. Parse the reply to extract the answer and actions.
            c. Apply rule_checks() to the answer.
            d. Apply llm_grade() to the answer.
            e. Compute a final blended score (40% rule-based + 60% LLM-based).
            f. Store the results including tokens, notes, and scores.
        5. Compute the mean final score across all prompts.
        6. Save the evaluation report as 'reports/style_eval.json'.

    Args:
        prompts (List[str]): List of input prompts to evaluate.
        logger (logging.Logger): Logger instance for debugging and diagnostics.

    Returns:
        dict: A dictionary containing:
            - 'mean_final': Mean blended score across prompts.
            - 'items': A list of per-prompt result dictionaries with keys:
                'prompt', 'answer', 'rule_score', 'llm_score', 'final',
                'notes', 'tokens'.
    """
    # Initialize LangGraph and system prompt
    system_prompt, parser = load_system_prompt()
    graph = setup_langgraph(system_prompt, parser)
    thread_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    graph_config = {"configurable": {"thread_id": thread_id}}

    # Initialize LLM for grading
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
        temperature=src.bot.config.TEMPERATURE,
        max_output_tokens=src.bot.config.MAX_OUTPUT_TOKENS,
        timeout=src.bot.config.REQUEST_TIMEOUT,
    )

    results = []
    for prompt in prompts:

        # Get bot response
        # The same logic as in "Chat Loop" of the bot.
        # Check special first (like /order 66)
        # if None, then consider it a normal message

        special_result = handle_special_command(graph, graph_config, prompt, logger)
        if special_result:
            reply, tokens, _, _ = special_result
        else:
            reply, tokens = process_message(graph, graph_config, prompt, logger)

        print(f"User: {prompt}, \nBOT REPLY:\n{reply}, \n{tokens=}\n\n")

        # Parse to check structure and extract answer
        answer, actions = parse_reply(reply)
        if answer is None:
            final_score = 0
            notes = "Failed to parse structured output"
            llm_score = Grade(score=0, notes=notes)
            rule_score = 0
        else:
            rule_score = rule_checks(answer)
            llm_score = llm_grade(answer, llm)
            final_score = int(0.4 * rule_score + 0.6 * llm_score.score)
            notes = llm_score.notes

        results.append(
            {
                "prompt": prompt,
                "answer": reply,
                "rule_score": rule_score,
                "llm_score": llm_score.score,
                "final": final_score,
                "notes": notes,
                "tokens": tokens,
            }
        )

    mean_final = round(statistics.mean(r["final"] for r in results), 2)
    out = {"mean_final": mean_final, "items": results}

    # Save report
    report_path = REPORTS / "style_eval.json"
    report_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return out


if __name__ == "__main__":
    import logging
    from src.bot.app import setup_logging

    setup_logging(logging.INFO)
    logger = logging.getLogger()

    eval_prompts = (
        (BASE / "data/eval_prompts.txt")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    report = eval_batch(eval_prompts, logger)
    print("Mean score:", report["mean_final"])
    print("Report saved:", REPORTS / "style_eval.json")
