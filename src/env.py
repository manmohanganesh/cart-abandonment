from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

def get_env_variable(key: str,default=None):
    value=os.getenv(key,default)

    if value is None:
        raise ValueError(f"Environment variable '{key}' is not set")
    return value