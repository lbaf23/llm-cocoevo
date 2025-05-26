from code_evaluator import evaluate_code, get_line_cov_feedback
from utils import read_jsonl, write_file, write_jsonl
from argparse import ArgumentParser
import time


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_file', type=str, default='data/leetcode_contest.jsonl')
    parser.add_argument('--env_type', type=str, default='func')
    parser.add_argument('--num_process', type=int, default=5)
    parser.add_argument('--total_time_limit', type=int, default=2)  # 2 for leetcode_contest
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    file_path = args.dataset_file
    data = []
    content = read_jsonl(file_path)
    print(len(content))

    for i in range(0, len(content)):
        if args.start < args.end and not args.start <= i < args.end:
            continue

        d = content[i]
        start_at = time.time()

        if args.env_type == 'func':
            tests = content[i]['tests']
            code = content[i]['solution']
            res = evaluate_code(code, tests, env_type='func', data_args={}, num_process=args.num_process, total_time_limit=args.total_time_limit, feedback=True)
        elif args.env_type == 'real_world_function':
            tests = d['test']
            code = d['solution']
            data_args = {
                'prompt_test': d['prompt_test'],
                'context_program': d['context_program'],
                'context_test_program': d['context_test_program'],
            }
            res = evaluate_code(code, tests, env_type='real_world_function', data_args=data_args, num_process=args.num_process, total_time_limit=args.total_time_limit)
            print(res)
            res2 = get_line_cov_feedback(code, tests, env_type='real_world_function', data_args=data_args, total_time_limit=args.total_time_limit)
            print(res2['feedback'])

        elif args.env_type == 'real_world_method':
            tests = [content[i]['tests']]
            code = content[i]['solution']
            data_args = {
                'start_line': content[i]['start_line'],
                'end_line': content[i]['end_line'],
                'test_prefix': content[i]['test_prefix'],
                'solution': content[i]['solution'],
                'program': content[i]['program'],
                'entry_point': content[i]['entry_point'],
            }
            res = evaluate_code(code, tests, env_type='real_world', data_args=data_args, num_process=args.num_process, total_time_limit=args.total_time_limit)
            print(res)
        else:
            raise NotImplementedError

        end_at = time.time()
        dur = (end_at - start_at)

        assert res['score'] == 1.0, res['feedbacks']
        print(f'{i}: passed {len(tests)}, {dur} s')
