import os
from dotenv import load_dotenv
from openai import OpenAI
import mlflow

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo2 - cost tracking")
mlflow.openai.autolog()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together" 
input_cost = 0.27
output_cost = 0.85

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
            "total_cost": (completion.usage.prompt_tokens / 1000000) * input_cost + (completion.usage.completion_tokens / 1000000) * output_cost,
        },
    )

    return completion.choices[0].message.content

print(llm_query("Explain LLMOps vs MLOps. Format like a LinkedIn Post.", model))