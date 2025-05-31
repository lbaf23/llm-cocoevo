from generators import CodeGenerator
from utils import append_jsonl, read_jsonl, create_or_clear_file, init_log
from tqdm import tqdm
from typing import Dict
from running_utils import load_env
import os


def Sampling(
        index: int,
        code_generator: CodeGenerator,
        data: Dict,
        result_file: str,
        log_file: str,
        run_config: Dict,
        env_type: str,
) -> None:
    """
    sampling
    """

    sampling_nums = run_config['sampling_nums']
    temperature = run_config['temperature']
    max_tokens = run_config['max_tokens']

    r = 0

    # load result
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
    else:
        create_or_clear_file(log_file)
        create_or_clear_file(result_file)

    td = tqdm(initial=r, total=sampling_nums)
    td.set_description(f'''[{index}]''')

    prompt = data['prompt']
    data_args = data['data_args']

    while r < sampling_nums:
        gen = code_generator.generate(
            prompt=prompt,
            env_type=env_type,
            init_method='default',
            data_args=data_args,
            max_tokens=max_tokens,
            temperature=temperature
        )
        item = {
            'r': r,
            'code': gen['code'],
            'output': gen['output'],
            'tokens_count': gen['tokens_count']
        }
        append_jsonl(result_file, item)
        td.update(1)
        r += 1


if __name__ == '__main__':
    env = load_env()

    dataset = env['dataset']
    model = env['model']
    args = env['args']
    log_dir = env['log_dir']
    result_dir = env['result_dir']
    config = env['config']
    run_type = env['run_type']

    assert run_type.startswith('sampling')

    code_generator = CodeGenerator(model)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset.get_data(i)

        # create files
        log_file = os.path.join(log_dir, f'log_{i}.log')
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        init_log(log_file)

        run_config = config[args.run_type]
        Sampling(
            index=i,
            code_generator=code_generator,
            data=data,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
            env_type=args.env_type,
        )
