from utils import load_config, create_dirs
import os
from code_datasets import CodeDataset
from code_models import ModelBase, model_factory
from typing import *
import torch
import argparse


def init_cuda():
    if torch.cuda.is_available:
        try:
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        except Exception:
            print('cuda emtpy cache failed.', flush=True)


def read_args(add_args: List[Dict] = []) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', type=str, default='sampling')
    parser.add_argument('--env_type', type=str, default='func')
    parser.add_argument('--config', type=str, default='config.json')

    parser.add_argument('--total_time_limit', type=float, default=2.0)
    parser.add_argument('--num_process', type=int, default=5, help='The number of parallel processes to use when evaluating a program.')

    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--base_url', type=str, default='')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    parser.add_argument('--load_models', action='store_true')

    parser.add_argument('--selected_ids', nargs='+', type=int, default=[])

    for a in add_args:
        name = a['name']
        a.pop('name')
        parser.add_argument(name, **a)
    return parser.parse_args()


def load_env(
        run_type: str = '',
        build_run_type: Callable = None,
        add_args: List[Dict] = [],
        load_models: bool = True,
) -> Dict[str, Union[
        argparse.Namespace,
        ModelBase,
        str,
        Dict[str, Any],
        CodeDataset,
        str,
        str
]]:
    args = read_args(add_args)

    if args.run_type != '':
        run_type = args.run_type
    elif build_run_type != None:
        run_type = build_run_type(args)
    assert run_type != '', '''run_type can't be none'''

    if args.load_models:
        load_models = args.load_models

    config = load_config(args.config)

    if load_models:
        init_cuda()
        extra_args = {}
        if args.api_key != '':
            extra_args['api_key'] = args.api_key
        if args.base_url != '':
            extra_args['base_url'] = args.base_url
        model = model_factory(**config['model'], **extra_args)
    else:
        model = None

    dataset = CodeDataset(**config['dataset'], selected_ids=args.selected_ids)

    result_dir = os.path.join(config['result_dir'], run_type)
    log_dir = os.path.join(config['log_dir'], run_type)
    create_dirs(result_dir)
    create_dirs(log_dir)

    if args.api_key != '':
        args.api_key = 'xxx'
    print(args)

    return {
        'args': args,
        'model': model,
        'run_type': run_type,
        'config': config,
        'dataset': dataset,
        'result_dir': result_dir,
        'log_dir': log_dir
    }
