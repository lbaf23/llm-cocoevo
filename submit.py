from utils import read_jsonl, append_jsonl, create_or_clear_file
from code_evaluator import evaluate_code
import os
from tqdm import tqdm
import json
from running_utils import load_env
import random
from typing import List


def get_k_codes(run_type: str, file_path: str, k: int) -> List[str]:
    content = read_jsonl(file_path)
    if run_type.startswith('sampling_filtering'):
        random.shuffle(content)
        content = sorted(content, key=lambda x: x['score'], reverse=True)
        return [c['code'] for c in content[ : k]]
    elif run_type.startswith('sampling'):
        codes = [c['code'] for c in content]
        return random.sample(codes, k)
    elif run_type.startswith('codet'):
        codes = []
        for c in content:
            codes += random.sample(c['codes'], 1)
        return codes[ : k]
    elif run_type.startswith('mbr_exec'):
        content = content[ : k]
        return [c['code'] for c in content]
    elif run_type.startswith('self_repair'):
        random.shuffle(content)
        content = sorted(content, key=lambda x: x['score'], reverse=True)
        codes = [c['code'] for c in content]
        return codes[ : k]
    elif run_type.startswith('coevo') or run_type.startswith('evolution'):
        codes = content[-1]['code_population']
        random.shuffle(codes)
        codes = sorted(codes, key=lambda x: x['fitness'], reverse=True)
        return [c['code'] for c in codes[ : k]]
    elif run_type.startswith('reflexion'):
        return [content[-1]['code']]
    else:
        raise NotImplementedError


def float2str(f):
    return str(round(f * 100, 5))


def to_str(res):
    return f'''{float2str(res['pass_rate'])} {res['passed_count']}/{res['total_count']}'''


if __name__ == '__main__':
    env = load_env(
        add_args=[
            {'name': '--k', 'type': int, 'default': 1},
            {'name': '--suffix', 'type': str, 'default': '-1'}
        ],
        load_models=False
    )

    dataset = env['dataset']
    config = env['config']
    args = env['args']
    run_type = env['run_type']

    result_dir = os.path.join(config['result_dir'], run_type)

    submit_result_file = os.path.join(result_dir, f'submit_k={args.k}{args.suffix}.jsonl')
    create_or_clear_file(submit_result_file)

    passed_count = 0
    total_count = 0

    passed_count_list = [0, 0, 0]
    total_count_list = [0, 0, 0]

    k = args.k

    for i in tqdm(dataset.data_range):
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset.get_data(i)

        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        if not os.path.exists(result_file):
            continue

        passed = False
        passed_code = ''
        codes = get_k_codes(run_type, result_file, k)
        for code in codes:
            if args.env_type == 'func':
                res = evaluate_code(
                    code=code,
                    tests=data['tests'],
                    evaluator_type=args.env_type,
                    num_process=args.num_process,
                    total_time_limit=args.total_time_limit
                )
            elif args.env_type == 'io':
                res = evaluate_code(
                    code=code,
                    tests=data['tests_original'],
                    evaluator_type=args.env_type,
                    num_process=args.num_process,
                    total_time_limit=args.total_time_limit,
                    test_main=data['test_main']
                )
            else:
                raise NotImplementedError(f'env_type={args.env_type} not supported')

            if res['score'] == 1.0:
                passed = True
                passed_code = code
                break

        append_jsonl(submit_result_file, {
            'index': i,
            'passed': passed,
            'passed_code': passed_code
        })

        if passed:
            passed_count += 1
        total_count += 1

        if args.env_type == 'func':
            if data['difficulty'].lower() == 'easy':
                total_count_list[0] += 1
                if passed:
                    passed_count_list[0] += 1
            elif data['difficulty'].lower() == 'medium':
                total_count_list[1] += 1
                if passed:
                    passed_count_list[1] += 1
            elif data['difficulty'].lower() == 'hard':
                total_count_list[2] += 1
                if passed:
                    passed_count_list[2] += 1
            else:
                raise NotImplementedError


    if args.env_type == 'func':
        res = {
            'pass_rate': passed_count / total_count,
            'passed_count': passed_count,
            'total_count': total_count,
            'easy': {
                'pass_rate': passed_count_list[0] / total_count_list[0],
                'passed_count': passed_count_list[0],
                'total_count': total_count_list[0]
            },
            'medium': {
                'pass_rate': passed_count_list[1] / total_count_list[1],
                'passed_count': passed_count_list[1],
                'total_count': total_count_list[1]
            },
            'hard': {
                'pass_rate': passed_count_list[2] / total_count_list[2],
                'passed_count': passed_count_list[2],
                'total_count': total_count_list[2]
            }
        }
        print(f'''{to_str(res)}, easy: {to_str(res['easy'])}, medium: {to_str(res['medium'])}, hard: {to_str(res['hard'])}''')

    else:
        res = {
            'pass_rate': passed_count / total_count,
            'passed_count': passed_count,
            'total_count': total_count
        }
        print(f'''{to_str(res)}''')

    append_jsonl(submit_result_file, res)
    print(json.dumps(res, indent=4), flush=True)

    print('>>> save to ', submit_result_file)
