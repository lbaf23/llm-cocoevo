from typing import List, Dict, Tuple
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import ast


def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def extract_asserts(output: str) -> List[str]:
    tests = []
    lines = output.split('\n')
    for l in lines:
        l = l.strip()
        if l.startswith('assert '):
            if is_syntax_valid(l):
                tests.append(l)
    return tests


def extract_code(output: str, sig: str = '```') -> str:
    code = extract_first_block(output, sig)
    lines = code.split('\n')
    code = ''
    i = 0
    while i < len(lines):
        if lines[i].startswith('def'):
            break
        i += 1
    
    i += 1
    
    while i < len(lines):
        if lines[i].startswith('assert') or lines[i].startswith('#') or lines[i].startswith('def'):
            break
        i += 1
    
    code = '\n'.join(lines[ : i])

    return code.strip()


def extract_first_block(output: str, sig = '```') -> str:
    content = output
    output = output.split('\n')
    block = ''
    i = 0
    while i < len(output):
        line = output[i]
        if line.startswith(sig):
            break
        i += 1
    i += 1

    while i < len(output):
        line = output[i]
        if line.startswith('```'):
            break
        
        block += line + '\n'
        i += 1
    block = block.strip()
    if block == '':
        block = content
    
    return block


@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def gpt_chat(
        client: OpenAI,
        model: str,
        messages: List[Dict],
        max_tokens: int = 1024,
        stop_strs: List[str] = [],
        temperature: float = 0.8
) -> Dict:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=stop_strs
    )

    return response.choices[0].message.content


class APIModels:
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
    ) -> str:
        return gpt_chat(
            client=self.client,
            model=self.model_path,
            messages=messages,
            max_tokens=max_tokens,
            stop_strs=stop_strs,
            temperature=temperature
        )
