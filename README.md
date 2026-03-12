<p align="center">
	<img src="assets/berrybrain.png" alt="berrybrain artwork" width="360" />
</p>

BerryBrain is a local AI hub for experimenting with Hugging Face model inference, tracking runs with MLflow, and organizing prompts and agents using LangChain. It helps you compare models, iterate on prompts, and keep reproducible experiment logs locally — and includes a Chainlit-hosted web UI for interactive demos and testing.

**Key components**
- **MLflow**: local experiment tracking and artifact storage (see `mlflow/`).
- **Hugging Face inference**: call different model endpoints via the HF router or provider-specific endpoints.
- **LangChain**: organize prompts, chains and agents that orchestrate model calls and pipelines.
- **Chainlit**: lightweight web UI to expose interactive agents and prompt experiments.

## Features
- Local MLflow server with autologging integration.
- Example multi-model client integration (`app.py`).
- Chainlit web UI for interactive testing and demos (optional scaffold).
- Planned folder structure for prompts and agents to support rapid iteration.

## Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Add your Hugging Face token to a `.env` file in the repo root:

```
HF_TOKEN=your_huggingface_token
```

4. Start the MLflow server (starts UI at http://127.0.0.1:5000):

```bash
./mlflow/start-mlflow.sh
```

5. Run the example application:

```bash
python app.py
```

6. (Optional) Run the Chainlit web UI or demos:

- The `demo/` folder is a small subfolder containing example Chainlit demos intended for colleagues; it's optional and for demonstration purposes only.
- From the repository root you can run a demo directly:

```bash
chainlit run demo/demo1-3.py -w
```

- Or change into the `demo/` folder and run the demo there:

```bash
cd demo
chainlit run demo1-3.py -w
```

Open the MLflow UI at http://127.0.0.1:5000 to inspect runs, metrics and artifacts, and open the Chainlit UI (default shown in the terminal) for interactive agent demos.

## Project layout (planned)
- `app.py` — example showing model calls and MLflow autologging.
- `mlflow/` — helper script and artifacts directory for the local MLflow server.
- `assets/` — images and UI assets.
- `prompts/` — prompt templates and tests (planned).
- `agents/` — LangChain agent implementations (planned).
- `chainlit/` — (optional) Chainlit app and UI assets (planned).

## Notes and recommendations
- Ensure the MLflow server is running before launching experiments so autologging captures runs.
- Swap model IDs in `app.py` for the models you want to compare; the example demonstrates calling multiple endpoints.
- Use `prompts/` to store canonical prompt templates and `agents/` to encapsulate orchestration and memory with LangChain.
- Use Chainlit to expose interactive demos and QA interfaces for prompt and model evaluation.

## Next steps
- Add a `prompts/` scaffold and a basic LangChain agent implementation.
- Scaffold a minimal Chainlit app under `chainlit/` that demonstrates model selection and logs results to MLflow.
- Add notebooks or scripts showing how to compare prompt-model combinations and record metrics in MLflow.