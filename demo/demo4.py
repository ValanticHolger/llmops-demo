import json
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.entities import SpanType
from openai import OpenAI

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo4 - tool usage")
mlflow.openai.autolog()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

MODEL_NAME = "Llama-4"

def load_model_settings(model_name: str) -> tuple[str, float, float]:
    with Path(__file__).with_name("models.json").open("r", encoding="utf-8") as f:
        models = json.load(f)

    model_config = models[model_name]
    return (
        model_config["provider_model"],
        model_config["input_cost"],
        model_config["output_cost"],
    )


@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(location: str) -> str:
    """Mock-Funktion: Wetter-API abfragen."""
    return f"{location}: 15°C und sonnig"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Ermittelt das aktuelle Wetter für einen bestimmten Ort.",
            "parameters": {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Die Stadt, z.B. Ludwigsburg, Deutschland",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

TOOL_REGISTRY = {
    "get_weather": get_weather,
}


def run_tool_calls(tool_calls, messages: list, tool_registry: dict) -> None:
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_fn = tool_registry.get(tool_name)
        if not tool_fn:
            continue

        # Tool-Calling Magic
        arguments = json.loads(tool_call.function.arguments or "{}")
        tool_result = tool_fn(**arguments)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result,
            }
        )


@mlflow.trace(span_type=SpanType.AGENT)
def llm_query_with_tools(prompt: str, model: str, input_cost: float, output_cost: float) -> str:
    span = mlflow.get_current_active_span()
    messages = [{"role": "user", "content": prompt}]

    total_input_tokens = 0
    total_output_tokens = 0

    while True:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        response_message = completion.choices[0].message
        total_input_tokens += completion.usage.prompt_tokens
        total_output_tokens += completion.usage.completion_tokens

        messages.append(response_message)

        if not response_message.tool_calls:
            final_response = response_message.content
            break

        run_tool_calls(response_message.tool_calls, messages, TOOL_REGISTRY)

    span.set_attribute(
        "mlflow.llm.cost",
        {
            "input_cost": (total_input_tokens / 1_000_000) * input_cost,
            "output_cost": (total_output_tokens / 1_000_000) * output_cost,
            "total_cost": (total_input_tokens / 1_000_000) * input_cost
            + (total_output_tokens / 1_000_000) * output_cost,
        },
    )

    return final_response


model, input_cost, output_cost = load_model_settings(MODEL_NAME)

llm_query_with_tools(
    "Wie ist das Wetter heute in Ludwigsburg?",
    model,
    input_cost,
    output_cost,
)