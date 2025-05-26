"""
CodeT

"""

from code_evaluator import evaluate_code
from typing import Tuple, List, Dict, Any
from utils import write_jsonl, get_unique_tests, get_codes, create_dirs
from running_utils import load_env
from tqdm import tqdm
import math
import os


def CodeT(
        index: int,
        codes: List[str],
        tests: List[str],
        result_file: str,
        env_type: str,
        data_args: Dict[str, Any],
        num_process: int,
        total_time_limit: int
) -> None:
    code_passed_tests = [[] for _ in range(len(codes))]
    """
    code i passed tests:
        code_passed_tests[i] = [test_index1, test_index2, ...]
    """

    for i, code in enumerate(tqdm(codes, desc=f'[{index}]')):
        res = evaluate_code(
            code=code,
            tests=tests,
            env_type=env_type,
            data_args=data_args,
            num_process=num_process,
            total_time_limit=total_time_limit
        )
        tests_index = []
        for j, r in enumerate(res['status']):
            if r == True:
                tests_index.append(j)
        code_passed_tests[i] = tests_index

    # agreement groups
    agreement = dict()
    """
    test_set : code_set
        agreement[tuple(tests_index)] = [code_index1, code_index2, ...]
    """
    for i, code in enumerate(codes):
        tests_index = tuple(code_passed_tests[i])
        if agreement.__contains__(tests_index):
            agreement[tests_index].append(i)
        else:
            agreement[tests_index] = [i]

    res = []
    for k in agreement.keys():
        codes_i = [codes[index] for index in agreement[k]]
        tests_i = [tests[index] for index in k]
        reward = math.sqrt(len(codes_i)) * len(tests_i)
        res.append({
            'reward': reward,
            'codes': codes_i,
            'tests': tests_i,
        })

    res = sorted(res, key=lambda x: x['reward'], reverse=True)
    write_jsonl(result_file, res)


if __name__ == '__main__':
    env = load_env(load_models=False)

    args = env['args']
    config = env['config']
    run_type = env['run_type']
    dataset = env['dataset']

    assert run_type.startswith('codet')

    run_config = config[run_type]
    max_codes = run_config['max_codes']
    max_tests_generations = run_config['max_tests_generations']
    max_tests_per_generation = run_config['max_tests_per_generation']

    codes_run_type = run_config['codes']
    tests_run_type = run_config['tests']

    result_dir = env['result_dir']
    codes_dir = os.path.join(result_dir.rstrip(run_type), codes_run_type)
    tests_dir = os.path.join(result_dir.rstrip(run_type), tests_run_type)
    create_dirs(result_dir)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        result_file = os.path.join(result_dir, f'result_{i}.jsonl')

        codes_file = os.path.join(codes_dir, f'result_{i}.jsonl')
        tests_file = os.path.join(tests_dir, f'result_{i}.jsonl')
        codes = get_codes(codes_file)
        codes = codes[ : max_codes]
        tests = get_unique_tests(tests_file, max_tests_generations, max_tests_per_generation)

        data = dataset.get_data(i)
        CodeT(
            index=i,
            codes=codes,
            tests=tests,
            result_file=result_file,
            env_type=args.env_type,
            data_args=data['data_args'],
            num_process=args.num_process,
            total_time_limit=args.total_time_limit
        )
