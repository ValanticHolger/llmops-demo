import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import mlflow
from mlflow.entities import Feedback
from mlflow.genai import evaluate, scorers
from mlflow.genai.datasets import search_datasets

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_BASE_URL"] = "https://router.huggingface.co/v1"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo5 - evaluation with judges")

APP_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together"
JUDGE_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together"

EVAL_DATASET_NAME = "linkedin-prompt-expectation"
PROMPT_URI = "prompts:/linkedin-post/2"

client = OpenAI()


def load_eval_dataset(dataset_name: str):
    escaped_name = dataset_name.replace("'", "\\'")
    matches = search_datasets(filter_string=f"name = '{escaped_name}'")

    return matches[0]


def run_judge(judge_prompt: str, name: str) -> Feedback:
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )

    raw = (completion.choices[0].message.content or "").strip()

    score = 0.5
    rationale = raw or "Empty judge response"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(match.group(0)) if match else {}

    if isinstance(parsed, dict):
        if "score" in parsed:
            try:
                score = float(parsed["score"])
            except (TypeError, ValueError):
                pass
        if "rationale" in parsed:
            rationale = str(parsed["rationale"])

    if score > 1.0:
        score = score / 100.0

    score = max(0.0, min(1.0, score))
    return Feedback(name=name, value=score, rationale=rationale)


eval_dataset = load_eval_dataset(EVAL_DATASET_NAME)
prompt_template = mlflow.genai.load_prompt(PROMPT_URI)

@mlflow.trace
def my_llm_app(prompt: str, model: str | None = None) -> str:
    _ = model
    rendered_prompt = prompt_template.format(question=prompt)

    completion = client.chat.completions.create(
        model=APP_MODEL,
        messages=[{"role": "user", "content": rendered_prompt}],
    )
    return completion.choices[0].message.content or ""


@scorers.scorer(name="Authenticity", aggregations=["mean"])
def authenticity_score(inputs=None, outputs=None, expectations=None, trace=None) -> Feedback:

    judge_prompt = (
        "You are a strict evaluator for authenticity and human voice in a LinkedIn-style professional text. "
        "Evaluate OUTPUT against INPUT and EXPECTATION. "
        "Treat EXPECTATION as a strong reference for quality and intent, but not as absolute ground truth. "
        "Allow valid alternative wording/structure if quality is comparable. "
        "Use the full score range and do not default to middle values. "
        "Do not quantize to 0.25 steps. Give a continuous score. "
        "Scoring intent: "
        "90-100 truly human, specific, credible perspective and concrete signals; "
        "70-89 mostly authentic with minor generic phrasing; "
        "40-69 mixed authenticity with clear template/corporate wording; "
        "0-39 robotic, spammy, hollow, or not believable. "
        "Only use 75 if evidence is exactly balanced. "
        "Return STRICT JSON ONLY: {\"score\": <number 0..100>, \"rationale\": \"<max 2 short sentences>\"}.\n\n"
        f"INPUT:\n{json.dumps(inputs, ensure_ascii=False)}\n\n"
        f"OUTPUT:\n{json.dumps(outputs, ensure_ascii=False)}\n\n"
        f"EXPECTATION:\n{json.dumps(expectations, ensure_ascii=False)}\n"
    )

    return run_judge(judge_prompt, name="Authenticity")


@scorers.scorer(name="Innovation", aggregations=["mean"])
def innovation_score(inputs=None, outputs=None, expectations=None, trace=None) -> Feedback:

    judge_prompt = (
        "You are a strict evaluator for innovation and originality in a LinkedIn-style professional text. "
        "Score whether OUTPUT provides fresh, non-obvious, useful thinking beyond generic advice. "
        "Treat EXPECTATION as a strong reference for depth and usefulness, but not as absolute ground truth. "
        "Do not require style or wording match; reward equally strong alternative ideas. "
        "Use the full score range and do not default to 75. "
        "Do not quantize to 0.25 steps. Give a continuous score. "
        "Scoring intent: "
        "90-100 distinctive angle plus actionable and non-trivial insight; "
        "70-89 useful with some freshness but partly conventional; "
        "40-69 mostly standard advice with little novelty; "
        "0-39 cliché, predictable, no meaningful new idea. "
        "Only use 75 if evidence is exactly balanced. "
        "Return STRICT JSON ONLY: {\"score\": <number 0..100>, \"rationale\": \"<max 2 short sentences>\"}.\n\n"
        f"OUTPUT:\n{json.dumps(outputs, ensure_ascii=False)}\n\n"
        f"EXPECTATION:\n{json.dumps(expectations, ensure_ascii=False)}\n"
    )

    return run_judge(judge_prompt, name="Innovation")


@scorers.scorer(name="Correctness", aggregations=["mean"])
def technical_correctness_score(inputs=None, outputs=None, expectations=None, trace=None) -> Feedback:

    judge_prompt = (
        "You are a strict evaluator for technical correctness and factual reliability. "
        "Score OUTPUT for factual accuracy, internal consistency, and responsible claim framing. "
        "Penalize overclaiming and unverifiable certainty. "
        "Treat EXPECTATION as a strong reference for what good coverage looks like, but not as absolute ground truth. "
        "Allow alternative but technically valid explanations if they are accurate and responsible. "
        "Use the full score range and do not default to 75. "
        "Do not quantize to 0.25 steps. Give a continuous score. "
        "Scoring intent: "
        "90-100 accurate, precise, nuanced, and responsibly framed; "
        "70-89 mostly correct with small imprecision; "
        "40-69 mixed correctness with notable ambiguity or errors; "
        "0-39 materially wrong, misleading, or deceptive. "
        "Only use 75 if evidence is exactly balanced. "
        "Return STRICT JSON ONLY: {\"score\": <number 0..100>, \"rationale\": \"<max 2 short sentences>\"}.\n\n"
        f"INPUT:\n{json.dumps(inputs, ensure_ascii=False)}\n\n"
        f"OUTPUT:\n{json.dumps(outputs, ensure_ascii=False)}\n\n"
        f"EXPECTATION:\n{json.dumps(expectations, ensure_ascii=False)}\n"
    )

    return run_judge(judge_prompt, name="Correctness")


with mlflow.start_run():
    results = evaluate(
        data=eval_dataset,
        predict_fn=my_llm_app,
        scorers=[
            authenticity_score,
            innovation_score,
            technical_correctness_score,
        ],
    )

print("\n=== Evaluation Abgeschlossen ===")
print("Ergebnisse:")
print(results.metrics)