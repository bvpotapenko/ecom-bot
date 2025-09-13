import json
import pathlib
import re
import statistics
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.app import load_system_prompt, setup_langgraph, process_message
from src import config

# Setup paths and environment
BASE = pathlib.Path(__file__).parent.parent
load_dotenv(BASE / ".env", override=True)
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)

# Simple rule-based checks
def rule_checks(text: str) -> int:
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
    score: int = Field(..., ge=0, le=100)
    notes: str

def setup_llm_grade_prompt():
    brand_name = config.SYSTEM_INSTRUCTION_TEMPLATE.split("{brand}")[0].strip()
    tone_persona = config.TONE.get("persona", "professional, friendly")
    tone_avoid = config.TONE.get("avoid", ["informal slang", "aggressive tone"])
    tone_must_include = config.TONE.get("must_include", ["clear", "polite"])
    
    return ChatPromptTemplate.from_messages([
        ("system", f"You are a strict reviewer ensuring responses align with the {brand_name} brand voice."),
        ("system", f"Tone: {tone_persona}. Avoid: {', '.join(tone_avoid)}. "
                   f"Must include: {', '.join(tone_must_include)}."),
        ("human", "Assistant response:\n{answer}\n\nProvide an integer score (0-100) and brief notes explaining the score.")
    ])

def llm_grade(text: str, llm) -> Grade:
    parser = llm.with_structured_output(Grade)
    prompt = setup_llm_grade_prompt()
    return (prompt | parser).invoke({"answer": text})

def eval_batch(prompts: List[str], logger) -> dict:
    # Initialize LangGraph and system prompt
    system_prompt = load_system_prompt()
    graph = setup_langgraph(system_prompt)
    thread_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    graph_config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize LLM for grading
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL_NAME", "gemini-2.5-flash"),
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        timeout=config.REQUEST_TIMEOUT,
    )

    results = []
    for prompt in prompts:
        # Get bot response
        reply, tokens = process_message(graph, graph_config, prompt, logger)
        
        # Evaluate
        rule_score = rule_checks(reply)
        llm_score = llm_grade(reply, llm)
        final_score = int(0.4 * rule_score + 0.6 * llm_score.score)
        
        results.append({
            "prompt": prompt,
            "answer": reply,
            "rule_score": rule_score,
            "llm_score": llm_score.score,
            "final": final_score,
            "notes": llm_score.notes,
            "tokens": tokens
        })
    
    mean_final = round(statistics.mean(r["final"] for r in results), 2)
    out = {"mean_final": mean_final, "items": results}
    
    # Save report
    report_path = REPORTS / "style_eval.json"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return out

if __name__ == "__main__":
    import logging
    from src.app import setup_logging
    args = setup_logging(logging.INFO)
    logger = logging.getLogger()
    
    eval_prompts = (BASE / "data/eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    report = eval_batch(eval_prompts, logger)
    print("Mean score:", report["mean_final"])
    print("Report saved:", REPORTS / "style_eval.json")