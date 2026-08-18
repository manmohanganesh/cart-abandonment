from openai import OpenAI

from src.env import get_env_variable


def generate_rag_answer(context, query):
    api_key = get_env_variable("GROQ_API_KEY")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    context_text = "\n".join(context)
    prompt = f"""
You are analyzing why users abandon carts.

Context from user reviews:
{context_text}

Question:
{query}

Answer in 2-3 concise bullet points.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content.strip()