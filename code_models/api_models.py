from typing import List, Dict, Tuple
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from .models import ModelBase


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def gpt_chat(
        client: OpenAI,
        model: str,
        messages: List[Dict],
        max_tokens: int = 1024,
        stop_strs: List[str] = [],
        temperature: float = 0.8
) -> Dict:
    args = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=stop_strs
    )

    response = client.chat.completions.create(**args)

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    return {
        'output': response.choices[0].message.content,
        'message': response.choices[0].message.model_dump(),
        'tokens_count': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    }


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def gpt_embed(
        client: OpenAI,
        model: str,
        input: str
) -> Dict:
    response = client.embeddings.create(
        input=input,
        model=model
    )
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens

    return {
        'output': response.data[0].embedding,
        'tokens_count': {
            'prompt_tokens': prompt_tokens,
            'total_tokens': total_tokens
        }
    }


class APIModels(ModelBase):
    def __init__(self, name: str, model_path: str = 'gpt-3.5-turbo', **args):
        self.name = name
        self.model_path = model_path
        self.client = OpenAI(**args)

    def generate_chat(
            self,
            messages: List[Dict],
            max_tokens: int = 1024,
            stop_strs: List[str] = [],
            temperature: float = 0.8
    ) -> Dict:
        return gpt_chat(
            client=self.client,
            model=self.model_path,
            messages=messages,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            temperature=temperature
        )
