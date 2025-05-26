from jsonl_utils import read_jsonl, append_jsonl
from file_utils import create_or_clear_file
import os
from tqdm import tqdm
import json
import random
from typing import List


def get_k_codes(file_path: str, k: int) -> List[str]:
    content = read_jsonl(file_path)
    random.shuffle(content)
    content = sorted(content, key=lambda i: i['score'], reverse=True)
    content = content[ : k]
    return [c['code'] for c in content]


def float2str(f):
    return str(round(f * 100, 5))


def to_str(res):
    return f'''{float2str(res['pass_rate'])} {res['passed_count']}/{res['total_count']}'''


from io import StringIO
import sys
from multiprocessing import Process, Queue
def evaluate(code: str, tests: List[str]) -> float:
    if len(tests) == 0:
        return 0.0

    def execute(code: str, tests: List[str], q: Queue):
        # Execute the code if no syntax errors
        exec_vars = {}
        errors = []
        code = code + '\n\n\n' + '\n'.join(tests)

        std_out = sys.stdout
        sys.stdout = StringIO()
        try:
            exec(code, exec_vars)
            q.put(1.0)
        except Exception as e:
            q.put(0.0)

        sys.stdout = std_out

    q = Queue()
    p = Process(target=execute, args=(code, tests, q,))
    p.start()
    p.join(2.0)

    try:
        score = q.get(block=False)
    except Exception:
        p.terminate()
        p.join()
        score = 0.0

    return score


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--suffix', type=str, default='-1')
    parser.add_argument('--k', type=int, default=1)

    parser.add_argument('--env_type', type=str, default='func')
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset_path)

    submit_result_file = os.path.join(args.save_dir, f'submit_k={args.k}{args.suffix}.jsonl')
    create_or_clear_file(submit_result_file)

    passed_count = 0
    total_count = 0

    passed_count_list = [0, 0, 0]
    total_count_list = [0, 0, 0]

    k = args.k

    for i in tqdm(range(0, len(dataset))):
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset[i]

        result_file = os.path.join(args.save_dir, f'result_{i}.jsonl')
        if not os.path.exists(result_file):
            continue

        passed = False
        passed_code = ''
        codes = get_k_codes(result_file, k)
        for code in codes:
            if args.env_type == 'func':
                score = evaluate(
                    code=code,
                    tests=data['tests'],
                )
            else:
                raise NotImplementedError

            if score == 1.0:
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

    res = {
        'pass_rate': passed_count / total_count,
        'passed_count': passed_count,
        'total_count': total_count,
    }
    append_jsonl(submit_result_file, res)
    print(json.dumps(res, indent=4), flush=True)

    print(f'''{to_str(res)}''')
    print('>>> save to ', submit_result_file)
