from argparse import ArgumentParser
from utils import read_jsonl, write_json
from code_evaluator import evaluate_code
import os
from code_datasets import  CodeDataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='result/leetcode_contest/qwen2.5-coder-32b/coevod_5')
    parser.add_argument('--env_type', type=str, default='func')
    parser.add_argument('--its', type=int, default=10)
    args = parser.parse_args()

    if args.env_type == 'func':
        dataset = CodeDataset('leetcode', 'data/leetcode_contest.jsonl')
    elif args.env_type == 'repo_exec':
        dataset = CodeDataset('repo_exec', 'data/repo_exec.jsonl')
    else:
        raise NotImplementedError

    result_dir = args.result_dir
    population_save_file = os.path.join(result_dir, 'results_test_population.json')
    offspring_save_file = os.path.join(result_dir, 'results_test_offspring.json')

    population_acc_list = [0 for _ in range(args.its)]
    offspring_acc_list = [0 for _ in range(args.its)]

    for i in range(80):
        data = dataset.get_data(i)
        code = data['solution']

        file_path = f'{args.result_dir}/result_{i}.jsonl'
        content = read_jsonl(file_path)

        population_acc_i = []
        offspring_acc_i = []
        for r, c in enumerate(content):
            # calculate test population acc
            test_population = c['test_population']
            tests = [t['test'] for t in test_population]
            res = evaluate_code(
                code=code,
                tests=tests,
                env_type=args.env_type,
                data_args=data['data_args']
            )
            acc_i = res['score']
            population_acc_list[r] += acc_i

            population_acc_i.append(acc_i)


            # calculate test offspring acc
            test_offspring = c['test_offspring']
            if type(test_offspring[0]) == str:
                tests = test_offspring
            else:
                tests = [t['test'] for t in test_offspring]

            res = evaluate_code(
                code=code,
                tests=tests,
                env_type=args.env_type,
                data_args=data['data_args']
            )
            acc_i = res['score']
            offspring_acc_list[r] += acc_i

            offspring_acc_i.append(acc_i)

        print(f'''[{i}]\n{population_acc_i}\n{offspring_acc_i}''')

    for i in range(args.its):
        population_acc_list[i] /= 80
        offspring_acc_list[i] /= 80

    write_json(population_save_file, {
        'acc_list': population_acc_list,
    })
    write_json(offspring_save_file, {
        'acc_list': offspring_acc_list
    })

    print(f'''\
=== population acc ===
{population_acc_list}

=== offspring acc ===
{offspring_acc_list}

save to {population_save_file}
save to {offspring_save_file}
''')
