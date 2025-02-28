"""
Sampling+Filtering

"""

from utils import read_jsonl, write_jsonl, create_dirs, get_unique_tests
from code_evaluator import evaluate_code
from tqdm import tqdm
import os
from running_utils import load_env


if __name__ == '__main__':
    env = load_env(load_models=False)

    dataset = env['dataset']
    args = env['args']
    config = env['config']
    run_type = env['run_type']
    result_dir = env['result_dir']

    assert run_type.startswith('sampling_filtering')

    run_config = config[run_type]
    max_codes = run_config['max_codes']
    max_tests_generations = run_config['max_tests_generations']
    max_tests_per_generation = run_config['max_tests_per_generation']

    codes_dir = os.path.join(result_dir.rstrip(run_type), run_config['codes'])
    tests_dir = os.path.join(result_dir.rstrip(run_type), run_config['tests'])
    create_dirs(result_dir)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        result_file = os.path.join(result_dir, f'result_{i}.jsonl')

        tests_file = os.path.join(tests_dir, f'result_{i}.jsonl')
        tests = get_unique_tests(tests_file, max_tests_generations, max_tests_per_generation)

        codes = read_jsonl(os.path.join(codes_dir, f'result_{i}.jsonl'))
        codes = codes[ : max_codes]

        for c in tqdm(codes, desc=f'[{i}]'):
            code = c['code']
            res = evaluate_code(
                code,
                tests,
                evaluator_type=args.env_type,
                num_process=args.num_process,
                total_time_limit=args.total_time_limit
            )
            c['score'] = res['score']
            c['status'] = res['status']

        write_jsonl(result_file, codes)
