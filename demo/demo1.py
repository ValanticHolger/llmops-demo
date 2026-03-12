import os
from dotenv import load_dotenv
from openai import OpenAI
import mlflow

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo1 - easy logging")
mlflow.openai.autolog()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together"

def llm_query(prompt: str, model: str) -> str:
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    return completion.choices[0].message.content

print(llm_query("Explain LLMOps vs MLOps.", model))