import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import mlflow

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo3 - prompt templates")
mlflow.openai.autolog()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

CONFIG_PATH = Path(__file__).with_name("models.json")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    models = json.load(f)

model_name = "GPT-OSS-120B"

model_config = models[model_name]
model = model_config["provider_model"]
input_cost = model_config["input_cost"]
output_cost = model_config["output_cost"]

@mlflow.trace
def llm_query(prompt: str, model: str) -> str:

    span = mlflow.get_current_active_span()
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    span.set_attribute(
        "mlflow.llm.cost",
        {
            "input_cost": (completion.usage.prompt_tokens / 1000000) * input_cost,
            "output_cost": (completion.usage.completion_tokens / 1000000) * output_cost,
            "total_cost": (completion.usage.prompt_tokens / 1000000) * input_cost 
                        + (completion.usage.completion_tokens / 1000000) * output_cost,
        },
    )

    return completion.choices[0].message.content

# Load prompt from MLflow
prompt = mlflow.genai.load_prompt("prompts:/simple-explanation/1")

print(llm_query(prompt.format(), model))