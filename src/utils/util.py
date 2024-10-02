import json
import os
from pathlib import Path


def set_api_key_environ(api_key_path: str) -> None:
    """
    Reads the API key from the given path and sets the environ variable.
    """
    with open(api_key_path) as io:
        keys = json.load(io)

    for key, value in keys.items():
        os.environ[key] = value