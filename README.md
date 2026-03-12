# LLMOps Spotlight

This repository is a hands-on resource for an internal talk on **LLMOps with MLflow**.

It demonstrates how to:
- instrument LLM calls,
- track traces and costs,
- use prompt templates,
- capture tool usage,
- run LLM-as-a-judge evaluations.

---

## What’s inside

- `demo/demo1.py` — basic LLM call + automatic MLflow tracing
- `demo/demo2.py` — adds token/cost tracking as trace attributes
- `demo/demo3.py` — uses prompt templates loaded from MLflow Prompt Registry
- `demo/demo4.py` — tool-calling flow (agent + tool spans)
- `demo/demo5.py` — evaluation pipeline with custom judge scorers
- `demo/demo6.py` — Blue Guardrails tracing
- `demo/demo7.py` — Blue Guardrails hallucination demo set (multiple trace cases)
- `mlflow/start-mlflow.sh` — local MLflow server startup script

---

## Prerequisites

- Python 3.10+
- A Hugging Face token with access to the routed models
- macOS/Linux shell

---

## Setup

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
pip install --no-cache-dir "mlflow[extras]"
```

3. Create a `.env` file in repo root

```env
HF_TOKEN=your_huggingface_token_here
```

---

## Start MLflow

From repo root:

```bash
./mlflow/start-mlflow.sh
```

MLflow UI will be available at:

- http://127.0.0.1:5000

---

## Run demos

Open a second terminal (keep MLflow server running in the first one), activate the same environment, then run:

```bash
source .venv/bin/activate
python demo/demo1.py
python demo/demo2.py
python demo/demo3.py
python demo/demo4.py
python demo/demo5.py
python demo/demo6.py
python demo/demo7.py
```

Recommended order is `demo1` → `demo7`.

---

## Demo storyline (talk flow)

1. **Observability first**: capture every LLM call as a trace.
2. **Cost awareness**: attach cost metadata to spans.
3. **Prompt management**: externalize prompts via MLflow.
4. **Agent/tool transparency**: trace local tool invocation.
5. **Quality loop**: evaluate outputs with multiple judge dimensions.

---

## Notes

- The scripts are intentionally lightweight and demo-focused.
- Model names and prices in the demos are examples for the talk context.
- If prompt/dataset registry entries are missing, create them in MLflow UI first (used in `demo3.py` and `demo5.py`).

---

## Troubleshooting

- **`HF_TOKEN` missing**: ensure `.env` exists and is loaded.
- **Cannot reach MLflow**: verify server is running on `127.0.0.1:5000`.
- **Prompt/Dataset not found**: confirm registry entries exist with expected names/versions.
- **Model access error**: check token permissions and provider/model availability.
