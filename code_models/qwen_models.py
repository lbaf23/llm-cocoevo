from .models import ModelBase
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, BitsAndBytesConfig
from .utils import CodeStoppingCriteria
import torch


class QwenModels(ModelBase):
    def __init__(self, name: str, model_path: str = 'Qwen/Qwen2.5-Coder-7B-Instruct', dtype: str = 'bf16', **args):
        self.name = name
        self.model_path = model_path

        model_args = {}
        if dtype == 'bf16':
            model_args['torch_dtype'] = torch.bfloat16
        elif dtype == 'fp16':
            model_args['torch_dtype'] = torch.float16
        elif dtype == 'int8':
            model_args['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif dtype == 'int4':
            model_args['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            attn_implementation='flash_attention_2',
            **model_args
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f'====== model {model_path} loaded. ======', flush=True)

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: List[str] = [], temperature: float = 0.8) -> str:
        stop_criteria = CodeStoppingCriteria([self.tokenizer.encode(i, add_special_tokens=False) for i in stop_strs])
        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            **input_ids,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stopping_criteria=StoppingCriteriaList([stop_criteria])
        )
        return self.tokenizer.decode(outputs[0])

    def generate_chat(
            self,
            messages: List[Dict],
            max_tokens: int = 1024,
            stop_strs: List[str] = [],
            temperature: float = 0.8
    ) -> Dict[str, Dict]:
        stop_criteria = CodeStoppingCriteria([self.tokenizer.encode(i, add_special_tokens=False) for i in stop_strs])
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors='pt', return_dict=True, add_generation_prompt=True).to(self.model.device)
        prompt_tokens = len(inputs.input_ids[0])
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stopping_criteria=StoppingCriteriaList([stop_criteria])
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
