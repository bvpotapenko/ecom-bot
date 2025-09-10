import os
import json
import re
import statistics
import logging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import yaml
from app import (
    run_chat_scenario,
    ConversationChain,
    load_system_prompt,
    setup_gemini,
    setup_logging,
)

BASE = os.path.dirname(os.path.abspath(__file__)) + "/.."
load_dotenv(BASE + "/.env")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(REPORTS, exist_ok=True)

# Setup logging
log_file = setup_logging(logging.INFO)  # Use same logging setup as app.py
logger = logging.getLogger()

# Load style
with open(os.path.join(BASE, "data/style_guide.yaml"), "r", encoding="utf-8") as f:
    STYLE = yaml.safe_load(f)


# Rule checks
def rule_checks(text: str) -> int:
    score = 100

    with open(os.path.join(BASE, "data/style_guide.yaml"), "r", encoding="UTF-8") as f:
        style = yaml.safe_load(f)

    if re.search(r"[\U0001F300-\U0001FAFF]", text):  # Emojis
        score -= 20
    if "!!!" in text:  # Excessive !!!
        score -= 10
    if len(text) > 600:  # Too long
        score -= 10

    # Additional checks from style['tone']['avoid']
    avoid_list = style["tone"]["avoid"]
    if "канцелярит: согласно вышеизложенному, осуществите" in avoid_list:
        forbidden_phrases = [
            "согласно вышеизложенному",
            "осуществите",
            "в соответствии с",
        ]
        for phrase in forbidden_phrases:
            if phrase.lower() in text.lower():
                score -= 15

    return max(score, 0)


# LLM grade model
class Grade(BaseModel):
    score: int = Field(ge=0, le=100)
    notes: str


setup_gemini()
LLM = ChatGoogleGenerativeAI(
    model=os.environ.get("MODEL_NAME", "gemini-2.5-flash"), temperature=0
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", f"Ты — строгий ревьюер соответствия голосу бренда {STYLE['brand']}"),
        (
            "system",
            f"Тон: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
            f"Обязательно: {', '.join(STYLE['tone']['must_include'])}.",
        ),
        (
            "human",
            "Ответ ассистента:\n{answer}\n\nДай целочисленный score 0..100 и краткие заметки почему.",
        ),
    ]
)


def llm_grade(text: str) -> Grade:
    chain = GRADE_PROMPT | LLM | JsonOutputParser(pydantic_object=Grade)
    return chain.invoke({"answer": text})


# Setup conversation for eval
llm = ChatGoogleGenerativeAI(model=os.environ.get("MODEL_NAME"))
memory = ConversationChain(llm=llm).memory  # Reuse from app
conversation = ConversationChain(llm=llm, memory=memory)


def eval_batch(scenarios: List[List[str]]) -> dict:  # Scenarios are multi-turn lists
    results = []
    for scenario in scenarios:
        replies, _ = run_chat_scenario(
            conversation, scenario, logging.getLogger(), reset_memory=True
        )
        for reply in replies:
            # Extract 'answer' from formatted reply (assume first part is answer)
            answer = reply.split("\nTone check:")[0].strip()
            rule = rule_checks(answer)
            g = llm_grade(answer)
            final = int(0.4 * rule + 0.6 * g.score)
            results.append(
                {
                    "scenario": scenario,
                    "answer": answer,
                    "rule_score": rule,
                    "llm_score": g.score,
                    "final": final,
                    "notes": g.notes,
                }
            )
    mean_final = round(statistics.mean(r["final"] for r in results), 2)
    if mean_final < 80:
        print("Warning: Mean score below 80!")
    out = {"mean_final": mean_final, "items": results}
    with open(os.path.join(REPORTS, "style_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    # Load eval prompts as single-turn scenarios; for multi-turn, group them
    eval_prompts = [
        line.strip()
        for line in open(
            os.path.join(BASE, "data/eval_prompts.txt"), "r", encoding="utf-8"
        ).readlines()
        if line.strip()
    ]
    scenarios = [
        [p] for p in eval_prompts
    ]  # Single-turn; adjust for multi<<<<<<<<<<<<<!!!!!!!!!!!!
    report = eval_batch(scenarios)
    print("Средний балл:", report["mean_final"])
    print("Отчёт:", os.path.join(REPORTS, "style_eval.json"))
