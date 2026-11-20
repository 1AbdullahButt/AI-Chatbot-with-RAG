import os
from groq import Groq

# Model name can be changed with a Space variable called GROQ_MODEL
# (Groq retired llama-3.1-70b-versatile, so we use a current model)
DEFAULT_MODEL = "openai/gpt-oss-120b"


def call_grok_api(prompt):
    # Get the API key from Hugging Face Space secrets
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it in the Space settings.")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    messages = [
        {
            "role": "system",
            "content": "You answer questions using only the document text given to you. "
                       "If the answer is not in the text, say you could not find it.",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(messages=messages, model=model)
    return response.choices[0].message.content
