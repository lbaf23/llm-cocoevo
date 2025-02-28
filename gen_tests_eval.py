"""
evaluate generated tests

"""

from utils import read_jsonl, write_jsonl, create_dirs
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

    run_config = config[run_type]

    tests_dir = os.path.join(result_dir.rstrip(run_type), run_config['tests'])
    create_dirs(result_dir)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset.get_data(i)
        solution = data['solution']

        result_file = os.path.join(result_dir, f'result_{i}.jsonl')

        tests_file = os.path.join(tests_dir, f'result_{i}.jsonl')
        content = read_jsonl(tests_file)
        tests_set = {}

        for c in tqdm(content, desc=f'[{i}]'):
            tests = [t['test'] for t in c['tests']]
            res = evaluate_code(
                code=solution,
                tests=tests,
                evaluator_type=args.env_type,
                num_process=args.num_process,
                total_time_limit=args.total_time_limit
            )
            status = res['status']
            for i in range(len(tests)):
                c['tests'][i]['correct'] = status[i]

        write_jsonl(result_file, content)
