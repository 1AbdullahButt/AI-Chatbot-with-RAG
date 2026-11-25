import os
from groq import Groq

# Model name can be changed with a Space variable called GROQ_MODEL
# (Groq retired llama-3.1-70b-versatile, so we use a current model)
DEFAULT_MODEL = "openai/gpt-oss-120b"


def chat_completion(messages, temperature=0.2):
    """Send messages to Groq. Returns (answer_text, total_tokens_used)."""
    # Get the API key from Hugging Face Space secrets
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it in the Space settings.")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return answer, tokens
