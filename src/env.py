import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def load_env():
    """Manually parses .env to handle Windows UTF-8/UTF-16 encoding issues."""
    if not ENV_PATH.exists():
        return

    # Read with utf-8-sig to automatically strip Windows BOM characters
    with open(ENV_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


def get_env_variable(key_name: str) -> str:
    load_env()
    value = os.getenv(key_name)
    if not value:
        raise ValueError(
            f"Missing environment variable: '{key_name}'. Looked in path: {ENV_PATH}"
        )
    return str(value).strip()