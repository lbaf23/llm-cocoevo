from argparse import ArgumentParser
from utils import read_jsonl, write_json
import os
from code_evaluator import  evaluate_code
from code_datasets import CodeDataset
from utils.evolution_utils import best_one
from tqdm import tqdm
import random


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='result/leetcode_contest/qwen2.5-coder-32b/coevod_5')
    parser.add_argument('--env_type', type=str, default='func')
    parser.add_argument('--suffix', type=str, default='-1')
    parser.add_argument('--its', type=int, default=10)
    parser.add_argument('--total_time_limit', type=float, default=2.0)
    parser.add_argument('--num_process', type=int, default=5)
    args = parser.parse_args()

    if args.env_type == 'func':
        dataset = CodeDataset('leetcode', 'data/leetcode_contest.jsonl')
    elif args.env_type == 'real_world_function':
        dataset = CodeDataset('real_world_function', 'data/real_world_function.jsonl')
    elif args.env_type == 'repo_exec':
        dataset = CodeDataset('repo_exec', 'data/repo_exec.jsonl')
    else:
        raise NotImplementedError

    save_file = os.path.join(args.result_dir, f'results_code_population{args.suffix}.json')
    result_dir = args.result_dir

    its = args.its

    passed_ids = [[] for _ in range(its)]
    passed_counts = [0 for _ in range(its)]
    scores = []
    passed_count_list = []
    total_count_list = []

    total_count = len(dataset.data_range)
    for i in tqdm(range(total_count)):
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        content = read_jsonl(result_file)
        data = dataset.get_data(i)

        for r in range(its):
            if len(content) > r:
                codes = content[r]['code_population']
            else:
                codes = content[-1]['code_population']

            random.shuffle(codes)
            best = best_one(codes)
            code = best['code']
            res = evaluate_code(
                code=code,
                tests=data['tests'],
                env_type=args.env_type,
                data_args=data['data_args'],
                num_process=args.num_process,
                total_time_limit=args.total_time_limit
            )
            scores.append(res['score'])
            passed_count_list.append(res['passed_count'])
            total_count_list.append(res['total_count'])
            if res['score'] == 1.0:
                passed_ids[r].append(i)
                passed_counts[r] += 1

    print(args.result_dir)
    print()
    result = []

    pass_rate = []
    for c in passed_counts:
        pass_rate.append(c / total_count)

    result = {
        'passed_ids': passed_ids,
        'passed_counts': passed_counts,
        'pass_rate': pass_rate,
        'scores': scores,
        'passed_count_list': passed_count_list,
        'total_count_list': total_count_list,
    }
    write_json(save_file, result)

    print(result['pass_rate'], flush=True)
