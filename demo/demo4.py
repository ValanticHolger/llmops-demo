import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import mlflow
from mlflow.entities import SpanType

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("demo4 - tool usage")
mlflow.openai.autolog()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together"
input_cost = 0.27
output_cost = 0.85

@mlflow.trace(span_type=SpanType.TOOL)
def get_weather(location: str) -> str:
    """Mock-Funktion: Wetter-API abfragen."""
    return location + ": 15°C und sonnig"

tools_schema = [
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
        }
    }
]


@mlflow.trace(span_type=SpanType.AGENT)
def llm_query_with_tools(prompt: str, model: str) -> str:

    span = mlflow.get_current_active_span()
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema,
    )

    response_message = completion.choices[0].message
    
    total_input_tokens = completion.usage.prompt_tokens
    total_output_tokens = completion.usage.completion_tokens


    if response_message.tool_calls:
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_weather":
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"--> Führe lokales Tool aus mit Argumenten: {arguments}")
                tool_result = get_weather(location=arguments.get("location"))
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                })

        second_completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        final_response = second_completion.choices[0].message.content
        
        total_input_tokens += second_completion.usage.prompt_tokens
        total_output_tokens += second_completion.usage.completion_tokens
        
    else:
        final_response = response_message.content

    span.set_attribute(
        "mlflow.llm.cost",
        {
            "input_cost": (total_input_tokens / 1_000_000) * input_cost,
            "output_cost": (total_output_tokens / 1_000_000) * output_cost,
            "total_cost": (total_input_tokens / 1_000_000) * input_cost + (total_output_tokens / 1_000_000) * output_cost,
        },
    )

    return final_response


print("\n--- ANTWORT ---")
print(llm_query_with_tools("Wie ist das Wetter heute in Ludwigsburg?", model))