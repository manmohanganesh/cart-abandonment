from openai import OpenAI, OpenAIError

from src.config import Config
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


def setup_groq(config: Config):
    """Initializes and returns the Groq OpenAI-compatible client."""
    recovery_llm = config.get("recovery.llm", {})
    api_key = get_env_variable("GROQ_API_KEY")
    client = OpenAI(
        base_url=recovery_llm.get("base_url", "https://api.groq.com/openai/v1"),
        api_key=api_key,
    )
    return client


def build_prompt(shap_dict, config: Config):
    recovery_config = config.get("recovery", {})
    llm_config = recovery_config.get("llm", {})
    prompts_config = recovery_config.get("prompts", {})

    strategy_types = prompts_config.get(
        "strategy_types",
        [
            "Discount-based (price sensitivity)",
            "Urgency-based (fear of missing out)",
            "Value-based (highlight product benefit)",
        ],
    )
    strategies_count = prompts_config.get("strategies_count", 3)
    message_max_lines = prompts_config.get("message_max_lines", 2)

    prompt = f"""
You are an expert in e-commerce conversion optimization.
A user is likely to abandon their cart.
Here are the key behavioral drivers:
{shap_dict}

Your task:
Generate {strategies_count} DIFFERENT recovery strategies:
"""
    for i, strategy in enumerate(strategy_types, 1):
        prompt += f"{i}. {strategy}\n"

    prompt += f"""
Rules:
- Each message must be 1–{message_max_lines} lines
- Be persuasive, natural, and human-like
- Do NOT sound generic
- Tailor message to the given behavior signals

Output format:
"""
    for i in range(1, strategies_count + 1):
        prompt += f"{i}. ...\n"

    prompt += "\nBehavioral Breakdown:\n"
    for feature, values in shap_dict.items():
        prompt += f"- {feature}: {values:.2f}\n"

    prompt += f"\nGenerate a short, natural, personalized recovery message ({message_max_lines} lines maximum)."
    prompt += "\nFocus on solving user concerns (price, engagement, etc.), not generic marketing."

    return prompt


def generate_message_llm(shap_dict, config: Config):
    recovery_llm = config.get("recovery.llm", {})
    try:
        client = setup_groq(config)
        prompt = build_prompt(shap_dict, config)

        response = client.chat.completions.create(
            model=recovery_llm.get("model", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": prompt}],
            temperature=recovery_llm.get("temperature", 0.8),
            max_tokens=recovery_llm.get("max_tokens", 300),
        )

        return response.choices[0].message.content.strip()

    except OpenAIError as e:
        print(f"LLM API failed: {e}")
        return generate_message(shap_dict)  # Fallback option

