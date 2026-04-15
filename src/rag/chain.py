from google import genai
from src.env import get_env_variable

def setup_llm():
    api_key = get_env_variable("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client

def generate_rag_answer(context,query):
    client = setup_llm()
    
    context_text = "\n".join(context)

    prompt = f"""
You are analyzing why users abandon carts.

Context from user reviews:
{context_text}

Question:
{query}

Answer in 2-3 concise bullet points.
"""
    response = client.models.generate_content(
        model = "models/gemini-flash-lite-latest",
        contents=prompt,
    )
    
    return response.text.strip()