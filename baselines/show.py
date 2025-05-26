from jsonl_utils import read_jsonl
import os
from argparse import  ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='result/leetcode_contest/qwen2.5-coder-32b')
    parser.add_argument('--run_type_list', nargs='+', default=[
        'codecot',
        'agentcoder',
        'intervenor'
    ])
    parser.add_argument('--k_list', nargs='+', default=[1])
    parser.add_argument('--suffix_list', nargs='+', default=['-1', '-2', '-3', '-4', '-5'])
    args = parser.parse_args()

    k_list = args.k_list
    suffix_list = args.suffix_list
    run_type_list = args.run_type_list
    result_dir = args.result_dir

    for k in k_list:
        print(f'=== k={k} ===')
        for run_type in run_type_list:
            pass_rate = 0
            passed_count = 0
            detailed_pass_rate = [0, 0, 0]
            detailed_passed_count = [0, 0, 0]
            for suffix in suffix_list:
                file_path = os.path.join(result_dir, run_type, f'submit_k={k}{suffix}.jsonl')
                content = read_jsonl(file_path)
                if len(content) == 0 or not content[-1].__contains__('pass_rate'):
                    raise NotImplementedError(f'File {file_path} does not contain pass_rate')

                pass_rate += content[-1]['pass_rate']
                passed_count += content[-1]['passed_count']

                passed_ids = []
                for i, c in enumerate(content[:-1]):
                    if c['passed']:
                        passed_ids.append(i)

                print(len(passed_ids), passed_ids)

            pass_rate /= len(suffix_list)
            passed_count /= len(suffix_list)

            detailed_pass_rate[0] /= len(suffix_list)
            detailed_pass_rate[1] /= len(suffix_list)
            detailed_pass_rate[2] /= len(suffix_list)

            detailed_passed_count[0] /= len(suffix_list)
            detailed_passed_count[1] /= len(suffix_list)
            detailed_passed_count[2] /= len(suffix_list)
            
            print(f'''\
#### {run_type}
{pass_rate * 100:.2f} {passed_count}/80''')
