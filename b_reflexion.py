"""
Reflexion

"""

from generators import CodeGenerator
from code_evaluator import evaluate_code
from typing import Dict, List, Any
from utils import read_jsonl, append_jsonl, print_log, create_or_clear_file, write_jsonl, get_unique_tests, get_codes, \
    create_dirs, init_log
from tqdm import tqdm
import os
from running_utils import load_env


def Reflexion(
        index: int,
        code_generator: CodeGenerator,
        data: Dict,
        tests: List[str],
        result_file: str,
        log_file: str,
        run_config: Dict,
        env_type: str,
        num_process: int,
        total_time_limit: float
) -> None:
    max_tokens = run_config['max_tokens']
    max_message_tokens = run_config['max_message_tokens']
    temperature = run_config['temperature']

    iterator_rounds = run_config['iterator_rounds']

    prompt = data['prompt']
    data_args = data['data_args']

    r = 0
    history = None
    item = None

    # load result
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
        item = result[-1]
        if len(result) > 1:
            history = result[-2]

        if item['score'] == 1.0:
            return
    else:
        create_or_clear_file(result_file)
        create_or_clear_file(log_file)

    td = tqdm(initial=r, total=iterator_rounds)
    td.set_description(f'''[{index}]''')

    # init
    while r < iterator_rounds:
        if r == 0:
            gen = code_generator.generate(
                prompt=prompt,
                env_type=env_type,
                data_args=data_args,
                max_tokens=max_tokens,
                temperature=temperature
            )
            code = gen['code']
            res = evaluate_code(
                code=code,
                tests=tests,
                env_type=env_type,
                data_args=data_args,
                num_process=num_process,
                total_time_limit=total_time_limit,
                feedback=True
            )
            item = {
                'r': r,
                'method': 'init',
                'code': code,
                'score': res['score'],
                'feedbacks': res['feedbacks'],
                'status': res['status'],
                'tokens_count': gen['tokens_count']
            }
        else:
            gen = code_generator.generate_reflexion(
                prompt=prompt,
                item=item,
                history=history,
                env_type=env_type,
                data_args=data_args,
                max_message_tokens=max_message_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            code = gen['code']
            res = evaluate_code(
                code=code,
                tests=tests,
                env_type=env_type,
                data_args=data_args,
                num_process=num_process,
                total_time_limit=total_time_limit,
                feedback=True
            )
            history = item
            item = {
                'r': r,
                'method': 'reflexion',
                'code': code,
                'reflection_message': gen['reflection_message'],
                'score': res['score'],
                'feedbacks': res['feedbacks'],
                'status': res['status'],
                'tokens_count': gen['tokens_count']
            }

        append_jsonl(result_file, item)
        td.update(1)
        r += 1

        if item['score'] == 1.0:
            break


if __name__ == '__main__':
    env = load_env(load_models=True)

    dataset = env['dataset']
    args = env['args']
    config = env['config']
    run_type = env['run_type']
    result_dir = env['result_dir']
    log_dir = env['log_dir']

    assert run_type.startswith('reflexion'), 'The run_type must start with reflexion'

    model = env['model']

    code_generator = CodeGenerator(model)

    run_config = config[run_type]

    max_tests_generations = run_config['max_tests_generations']
    max_tests_per_generation = run_config['max_tests_per_generation']

    tests_dir = os.path.join(result_dir.rstrip(run_type), run_config['tests'])
    create_dirs(result_dir)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        log_file = os.path.join(log_dir, f'log_{i}.log')

        init_log(log_file)

        tests_file = os.path.join(tests_dir, f'result_{i}.jsonl')
        tests = get_unique_tests(tests_file, max_tests_generations, max_tests_per_generation)

        Reflexion(
            index=i,
            code_generator=code_generator,
            data=dataset.get_data(i),
            tests=tests,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
            env_type=args.env_type,
            num_process=args.num_process,
            total_time_limit=args.total_time_limit
        )
