from openai import OpenAI

from src.config import Config
from src.env import get_env_variable


def generate_rag_answer(context, query: str, config: Config):
    rag_llm = config.get("rag.llm", {})
    api_key = get_env_variable("GROQ_API_KEY")

    client = OpenAI(
        base_url=rag_llm.get("base_url", "https://api.groq.com/openai/v1"),
        api_key=api_key,
    )

    context_text = "\n".join(context)
    prompt = f"""
{rag_llm.get("prompts.rag_system", "You are analyzing why users abandon carts.")}

Context from user reviews:
{context_text}

Question:
{query}

{rag_llm.get("prompts.rag_answer_format", "Answer in 2-3 concise bullet points.")}
"""

    response = client.chat.completions.create(
        model=rag_llm.get("model", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=rag_llm.get("temperature", 0.7),
        max_tokens=rag_llm.get("max_tokens", 500),
    )

    return response.choices[0].message.content.strip()