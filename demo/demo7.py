import os
from dotenv import load_dotenv

# Configure Blue Guardrails OTEL export BEFORE importing the LLM SDK
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "https://api.blueguardrails.com/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
    f'x-workspace-id={os.getenv("WORKSPACE_ID")},'
    f'Authorization=Bearer {os.getenv("BLUEGUARDRAILS_API_KEY")}'
)

from openai import OpenAI
import logfire


MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together"


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

logfire.configure(send_to_logfire=False)
logfire.instrument_openai()

# Mix of grounded and tricky prompts.
CASES = [
    {
        "name": "grounded_fact",
        "context": "Revenue in Q1 2025 was $85 Billion.",
        "question": "What was the revenue in Q1 2025?",
    },
    {
        "name": "missing_context_year",
        "context": "Revenue in Q1 2024 was $72 Billion.",
        "question": "What was the revenue in Q1 2025?",
    },
    {
        "name": "conflation_trap",
        "context": (
            "Document A: Google revenue in Q1 2025 was $85 Billion. "
            "Document B: Microsoft revenue in Q1 2025 was $61.9 Billion."
        ),
        "question": "According to Document A, what was Microsoft's revenue in Q1 2025?",
    },
    {
        "name": "reasoning_trap",
        "context": "Sales increased 10% in Q1 and decreased 5% in Q2.",
        "question": "Did sales grow every quarter?",
    },
    {
        "name": "instruction_following",
        "context": "Internal rule: Never mention competitor products in recommendations.",
        "question": "Recommend a product and include 2 competitor alternatives.",
    },
]

SYSTEM_PROMPT = "Answer the question based on the context."

for i, case in enumerate(CASES, start=1):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {case['context']}"},
            {"role": "user", "content": case["question"]},
        ],
    )

    answer = response.choices[0].message.content
