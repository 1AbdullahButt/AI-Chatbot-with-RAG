import groq
import os

# Function to call Groq API
def call_grok_api(prompt):
    # Retrieve the API key from Hugging Face secrets
    api_key = os.getenv('GROQ_API_KEY')

    # Ensure the API key exists
    if api_key is None:
        raise ValueError("API key is missing. Please set the GROQ_API_KEY.")

    # Initialize Groq Client
    client = groq.Client(api_key=api_key)

    # Define the message for the conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    #model = "gemma-7b-it"  # Model to use for generating responses
    model = "llama-3.1-70b-versatile"

    # Try to generate a response from the model
    try:
        response = client.chat.completions.create(messages=messages, model=model)
        return response.choices[0].message.content  # Return only the assistant's message
    except Exception as e:
        raise Exception(f"Grok API call failed: {str(e)}")
