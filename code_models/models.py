from typing import List, Dict


class ModelBase():
    def __init__(self, name: str, model_path: str, **args):
        raise NotImplementedError

    def generate_chat(
            self,
            messages: List[Dict],
            tools: List[Dict] = [],
            stop_strs: List[str] = [],
            temperature: float = 0.8,
            max_tokens: int = 1024
    ) -> Dict:
        raise NotImplementedError

    def generate(
            self,
            prompt: str,
            stop_strs: List[str] = [],
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict:
        raise NotImplementedError