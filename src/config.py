from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    def __init__(self, config_dict: dict):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def to_dict(self) -> dict:
        return self._config


def load_config(config_path: str = "configs/config.yaml") -> Config:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)