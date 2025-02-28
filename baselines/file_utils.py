import json
import os
from typing import Dict


def write_file(file_path: str, content: str | Dict):
    if isinstance(content, dict):
        content = json.dumps(content)
    with open(file_path, 'w') as file:
        file.write(content + '\n')


def create_dirs(file_dir: str):
    try:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    except Exception:
        pass


def exists_file(file_path: str) -> bool:
    return os.path.exists(file_path)


def read_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def create_or_clear_file(file_path: str):
    with open(file_path, 'w'):
        pass


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config
