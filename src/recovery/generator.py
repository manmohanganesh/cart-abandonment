from google import genai
from src.env import get_env_variable

def generate_message(shap_dict):
    """
    shap_dict : dictionary of feature -> contribution
    """
    message = "Hey! "

    if shap_dict.get("price_sensitivity",0)>0:
        message += "We noticed you might be comparing prices. "
        message += "Here's a small discount to help you decide "

    if shap_dict.get("scroll_depth",0)<0:
        message += "Take another look—you might find something you like. "

    message += "Complete your purchase now and enjoy the offer! "

    return message

def setup_gemini(config):
    api_key = get_env_variable("GEMINI_API_KEY")
    client=genai.Client(api_key=api_key)

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
    """
    
    for feature, values in shap_dict.items():
        prompt += f"- {feature}: {values :.2f}\n"
    
    prompt += "\nGenerate a short, natural, personalized recovery message (2 lines maximum)."
    prompt += "\nFocus on solving user concerns (price, engagement, etc.), not generic marketing."

    return prompt

def generate_message_llm(shap_dict,config):
    try:
        client = setup_gemini(config)
        
        prompt = build_prompt(shap_dict)
        model_name = config["llm"]["model"]

        response = client.models.generate_content(
            model=model_name,
            contents=prompt)

        return response.text.strip()
    except Exception as e:
        print(f"LLM failed:{e}")
        return generate_message(shap_dict) #fallback option