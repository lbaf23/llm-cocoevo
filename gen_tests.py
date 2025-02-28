"""
generate tests

"""

from generators import TestGenerator
from utils import append_jsonl, read_jsonl, create_or_clear_file, init_log
from tqdm import tqdm
from typing import Dict
from running_utils import load_env
import os


def GenTests(
        index: int,
        test_generator: TestGenerator,
        data: Dict,
        result_file: str,
        log_file: str,
        run_config: Dict,
) -> None:
    """
    sampling tests
    """

    generations = run_config['generations']
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

    td = tqdm(initial=r, total=generations)
    td.set_description(f'''[{index}]''')

    prompt = data['prompt']
    entry_point = data['entry_point']

    while r < generations:
        gen = test_generator.generate(
            prompt=prompt,
            entry_point=entry_point,
            max_tokens=max_tokens,
            temperature=temperature
        )

        all_tests = [{'test': t} for t in gen['tests']]
        append_jsonl(result_file, {
            'r': r,
            'tests': all_tests,
        })
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

    test_generator = TestGenerator(model)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset.get_data(i)

        # create files
        log_file = os.path.join(log_dir, f'log_{i}.log')
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        init_log(log_file)

        run_config = config[args.run_type]
        GenTests(
            index=i,
            test_generator=test_generator,
            data=data,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
        )
