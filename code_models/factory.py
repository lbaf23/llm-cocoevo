from .models import ModelBase
from .api_models import APIModels
from .llama_models import LlamaModels
from .deepseek_models import DeepseekModels
from .qwen_models import QwenModels


def model_factory(
        model_type: str,
        name: str,
        model_path: str,
        **args
) -> ModelBase:
    if model_type == 'llama':
        model = LlamaModels(name, model_path, **args)
    elif model_type == 'api':
        model = APIModels(name, model_path, **args)
    elif model_type == 'qwen':
        model = QwenModels(name, model_path, **args)
    elif model_type == 'deepseek':
        model = DeepseekModels(name, model_path, **args)
    else:
        raise NotImplementedError

    print(f'{model_type}: {model_path} loaded.', flush=True)
    return model
