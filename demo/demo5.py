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

client = OpenAI()
APP_MODEL = "Qwen/Qwen3.5-397B-A17B:novita"
JUDGE_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together"

eval_dataset = search_datasets(filter_string="name = 'simple-expl-exp'")[0]
prompt_template = mlflow.genai.load_prompt("prompts:/linkedin-post/1")


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

    score = max(0.0, min(1.0, score))
    return Feedback(name=name, value=score, rationale=rationale)


@mlflow.trace
def my_llm_app(prompt: str, model: str | None = None) -> str:

    _ = model
    rendered_prompt = prompt_template.format(question=prompt)
    
    completion = client.chat.completions.create(
        model=APP_MODEL,
        messages=[{"role": "user", "content": rendered_prompt}],
    )
    return completion.choices[0].message.content


@scorers.scorer(name="Authenticity", aggregations=["mean"])
def authenticity_score(inputs=None, outputs=None, expectations=None, trace=None) -> Feedback:

    judge_prompt = (
        "You are a STRICT evaluator for authenticity and voice for professional texts. "
        "Score how authentic, human, and credible the OUTPUT feels for a professional text based on INPUT context. "
        "Use holistic judgment instead of fixed penalties. "
        "Rubric: "
        "1.00 = sounds genuinely human, concrete perspective, believable experience signals, no corporate fluff; "
        "0.75 = mostly authentic but some generic phrasing; "
        "0.50 = mixed authenticity, noticeable template-like language; "
        "0.25 = mostly generic/marketing speak; "
        "0.00 = clearly robotic, spammy, or inauthentic. "
        "Consider: specificity of experience, believable tone, and avoidance of buzzword-only language. Clamp to [0,1]. "
        "Return STRICT JSON ONLY: {\"score\": <float 0..1>, \"rationale\": \"<max 2 short sentences>\"}.\n\n"
        f"INPUT:\n{json.dumps(inputs, ensure_ascii=False)}\n\n"
        f"OUTPUT:\n{json.dumps(outputs, ensure_ascii=False)}\n\n"
    )

    return run_judge(judge_prompt, name="Authenticity")


@scorers.scorer(name="Innovation", aggregations=["mean"])
def innovation_score(inputs=None, outputs=None, expectations=None, trace=None) -> Feedback:

    judge_prompt = (
        "You are a STRICT evaluator for innovation and originality in a professional text. "
        "Score whether OUTPUT offers fresh thinking, non-obvious insights, and a distinctive angle. "
        "Use EXPECTATION only as optional context, not as a style-copy target. "
        "Use holistic judgment instead of fixed penalties. "
        "Rubric: 1.00 = clearly original angle with actionable, non-trivial insight; "
        "0.75 = useful and somewhat fresh, but partly conventional; "
        "0.50 = mostly standard advice with limited novelty; "
        "0.25 = cliché-heavy and predictable; "
        "0.00 = fully generic with no meaningful new idea. "
        "Consider: novelty, depth of reasoning, and practical usefulness for a professional audience. Clamp to [0,1]. "
        "Return STRICT JSON ONLY: {\"score\": <float 0..1>, \"rationale\": \"<max 2 short sentences>\"}.\n\n"
        f"OUTPUT:\n{json.dumps(outputs, ensure_ascii=False)}\n\n"
        f"EXPECTATION:\n{json.dumps(expectations, ensure_ascii=False)}\n"
    )

    return run_judge(judge_prompt, name="Innovation")


@scorers.scorer(name="Correctness", aggregations=["mean"])
def technical_correctness_score(inputs=None, outputs=None, expectations=None, trace=None) -> Feedback:

    judge_prompt = (
        "You are a STRICT evaluator for technical correctness and factual reliability in a professional text. "
        "Score whether OUTPUT is technically accurate, internally consistent, and responsibly framed. "
        "Check claims against common professional/engineering reality; penalize overclaiming. "
        "Use holistic judgment instead of fixed penalties. "
        "Rubric: 1.00 = technically correct, precise wording, no misleading claims; "
        "0.75 = mostly correct with minor imprecision; "
        "0.50 = mixed correctness with notable inaccuracies or ambiguity; "
        "0.25 = several incorrect or misleading claims; "
        "0.00 = fundamentally wrong or deceptive. "
        "Consider: factual accuracy, nuance under uncertainty, and whether claims are framed responsibly. "
        "Use EXPECTATION only as optional context. Clamp to [0,1]. "
        "Return STRICT JSON ONLY: {\"score\": <float 0..1>, \"rationale\": \"<max 2 short sentences>\"}.\n\n"
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