import os
from dotenv import load_dotenv
from openai import OpenAI
import logfire

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

traces_endpoint = 'https://api.blueguardrails.com/v1/traces'
os.environ['OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'] = traces_endpoint
os.environ['OTEL_EXPORTER_OTLP_HEADERS'] = (
    f'x-workspace-id={os.getenv("WORKSPACE_ID")},'
    f'Authorization=Bearer {os.getenv("BLUEGUARDRAILS_API_KEY")}'
)

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)
logfire.configure(send_to_logfire=False)
logfire.instrument_openai()

response = client.chat.completions.create(
    model='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:together',
    messages=[
        {'role': 'system', 'content': 'Answer the question based on the context.'},
        {'role': 'user', 'content': 'Context: Revenue in Q1 was $85 Billion.'},
        {'role': 'user', 'content': 'What was Google\'s revenue in Q1 2025?'},
    ]
)