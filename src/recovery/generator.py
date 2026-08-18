from openai import OpenAI
from src.env import get_env_variable


def generate_message(shap_dict):
    """Fallback rule-based generator if LLM service is unreachable.

    shap_dict : dictionary of feature -> contribution
    """
    message = "Hey! "

    if shap_dict.get("price_sensitivity", 0) > 0:
        message += "We noticed you might be comparing prices. "
        message += "Here's a small discount to help you decide "

    if shap_dict.get("scroll_depth", 0) < 0:
        message += "Take another look—you might find something you like. "

    message += "Complete your purchase now and enjoy the offer! "

    return message


def setup_groq():
    """Initializes and returns the Groq OpenAI-compatible client."""
    api_key = get_env_variable("GROQ_API_KEY")
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    return client


def build_prompt(shap_dict):
    prompt = f"""
You are an expert in e-commerce conversion optimization.
A user is likely to abandon their cart.
Here are the key behavioral drivers:
{shap_dict}

Your task:
Generate 3 DIFFERENT recovery strategies:
1. Discount-based (price sensitivity)
2. Urgency-based (fear of missing out)
3. Value-based (highlight product benefit)

Rules:
- Each message must be 1–2 lines
- Be persuasive, natural, and human-like
- Do NOT sound generic
- Tailor message to the given behavior signals

Output format:
1. ...
2. ...
3. ...

Behavioral Breakdown:
"""
    for feature, values in shap_dict.items():
        prompt += f"- {feature}: {values:.2f}\n"

    prompt += "\nGenerate a short, natural, personalized recovery message (2 lines maximum)."
    prompt += "\nFocus on solving user concerns (price, engagement, etc.), not generic marketing."

    return prompt


def generate_message_llm(shap_dict, config=None):
    try:
        client = setup_groq()
        prompt = build_prompt(shap_dict)

        # Uses the active Groq production model
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"LLM failed: {e}")
        return generate_message(shap_dict)  # Fallback option