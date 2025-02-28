from .models import ModelBase
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from typing import Dict, List
import torch
from .utils import CodeStoppingCriteria


class DeepseekModels:
    def __init__(self, name: str, model_path: str = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', **args):
        self.name = name
        self.model_path = model_path

        self.max_length = 8192
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def generate_chat(self, messages: List[Dict], max_tokens: int = 1024, stop_strs: List[str] = [], temperature: float = 0.2) -> Dict:
        stop_criteria = CodeStoppingCriteria([self.tokenizer(i, add_special_tokens=False).input_ids for i in stop_strs])
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(self.model.device)
        prompt_tokens = len(inputs.input_ids[0])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, self.max_length - len(inputs.input_ids)),
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output = outputs[0][len(inputs.input_ids[0]) : ]
        completion_tokens = len(output)
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        return {
            'output': output,
            'tokens_count': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens
            }
        }
