from generators import CodeGenerator
from code_evaluator import evaluate_code
from typing import Dict, List
from utils import read_jsonl, append_jsonl, create_or_clear_file, get_unique_tests, get_codes, create_dirs, init_log
from tqdm import tqdm
import os
from running_utils import load_env


def SelfRepair(
        index: int,
        code_generator: CodeGenerator,
        data: Dict,
        codes: List[str],
        tests: List[str],
        result_file: str,
        log_file: str,
        run_config: Dict,
        env_type: str,
        num_process: int,
        total_time_limit: float
) -> None:
    max_tokens = run_config['max_tokens']
    temperature = run_config['temperature']

    init_nums = run_config['init_nums']

    assert len(codes) >= init_nums, f'len(codes) >= init_nums'
    codes = codes[ : init_nums]

    prompt = data['prompt']

    r = 0
    init_items = []

    # load result
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
        init_items = result[ : init_nums]
    else:
        create_or_clear_file(result_file)
        create_or_clear_file(log_file)

    td = tqdm(initial=r, total=init_nums * 2)
    td.set_description(f'''[{index}]''')

    # init
    while r < init_nums:
        code = codes[r]
        res = evaluate_code(
            code=code,
            tests=tests,
            evaluator_type=env_type,
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
        }
        init_items.append(item)
        append_jsonl(result_file, item)
        td.update(1)
        r += 1


    # repair
    while r < init_nums * 2:
        parent = init_items[r - init_nums]
        if parent['score'] < 1.0:
            code = parent['code']
            test_feedback = parent['feedbacks'][0]['message']
            gen = code_generator.generate_repair(
                prompt=prompt,
                code=code,
                test_feedback=test_feedback,
                max_tokens=max_tokens,
                temperature=temperature
            )
            code = gen['code']
            res = evaluate_code(
                code=code,
                tests=tests,
                evaluator_type=env_type,
                num_process=num_process,
                total_time_limit=total_time_limit,
                feedback=True
            )

            item = {
                'r': r,
                'method': 'repair',
                'parent_index': r - init_nums,
                'code': code,
                'output': gen['output'],
                'score': res['score'],
                'feedbacks': res['feedbacks'],
                'status': res['status']
            }
        else:
            item = parent

        append_jsonl(result_file, item)
        td.update(1)
        r += 1


if __name__ == '__main__':
    env = load_env(load_models=True)

    dataset = env['dataset']
    args = env['args']
    config = env['config']
    run_type = env['run_type']
    result_dir = env['result_dir']
    log_dir = env['log_dir']

    assert run_type.startswith('self_repair'), f'run_type should start with self_repair'

    model = env['model']

    code_generator = CodeGenerator(model)

    run_config = config[run_type]

    max_tests_generations = run_config['max_tests_generations']
    max_tests_per_generation = run_config['max_tests_per_generation']

    codes_dir = os.path.join(result_dir.rstrip(run_type), run_config['codes'])
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

        codes_file = (os.path.join(codes_dir, f'result_{i}.jsonl'))
        codes = get_codes(codes_file)

        SelfRepair(
            index=i,
            code_generator=code_generator,
            data=dataset.get_data(i),
            codes=codes,
            tests=tests,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
            env_type=args.env_type,
            num_process=args.num_process,
            total_time_limit=args.total_time_limit
        )
