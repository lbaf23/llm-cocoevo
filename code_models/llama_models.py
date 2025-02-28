from .models import ModelBase
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, BitsAndBytesConfig
from .utils import CodeStoppingCriteria
import torch


class LlamaModels(ModelBase):
    def __init__(self, name: str, model_path: str = 'meta-llama/Llama-3.1-8B-Instruct', dtype: str = 'bf16', **args):
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

        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', **model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f'====== model {model_path} loaded. ======', flush=True)

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: List[str] = [], temperature: float = 0.8) -> str:
        stop_criteria = CodeStoppingCriteria([self.tokenizer(i, add_special_tokens=False).input_ids for i in stop_strs])
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stopping_criteria=StoppingCriteriaList([stop_criteria])
        )
        output = outputs[0][len(inputs.input_ids[0]) : ]
        return self.tokenizer.decode(output, skip_special_tokens=True)

    eot = '<|eot_id|>'
    start_header_id = '<|start_header_id|>'
    end_header_id = '<|end_header_id|>'

    def generate_chat(self, messages: List[Dict], max_tokens: int = 1024, stop_strs: List[str] = [], temperature: float = 0.8) -> str:
        stop_criteria = CodeStoppingCriteria([self.tokenizer(i, add_special_tokens=False).input_ids for i in stop_strs])
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            stopping_criteria=StoppingCriteriaList([stop_criteria])
        )
        output = outputs[0][len(inputs.input_ids[0]) : ]
        output = self.tokenizer.decode(output, skip_special_tokens=False)

        output = output.rstrip(self.eot)
        if output.__contains__(self.end_header_id):
            output = output[output.index(self.end_header_id) + len(self.end_header_id) : ].lstrip()
        return output
