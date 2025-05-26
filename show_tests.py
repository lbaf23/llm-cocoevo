from utils import read_jsonl, get_unique_tests, write_json
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='result/leetcode_contest/llama-3.1-70b')
    parser.add_argument('--run_type', type=str, default='gen_tests_eval')
    parser.add_argument('--length', type=int, default=80)
    parser.add_argument('--max_tests_generations', type=int, default=10)
    parser.add_argument('--max_tests_per_generation', type=int, default=10)
    args = parser.parse_args()

    result_dir = args.result_dir
    run_type = args.run_type
    max_tests_generations = args.max_tests_generations
    max_tests_per_generation = args.max_tests_per_generation

    save_file = os.path.join(result_dir, run_type, 'result_test.json')

    passed_count = 0
    total_count = 0
    for i in range(args.length):
        file_path = os.path.join(result_dir, run_type, f'result_{i}.jsonl')
        tests = get_unique_tests(
            file_path=str(file_path),
            max_tests_generations=max_tests_generations,
            max_tests_per_generation=max_tests_per_generation,
            returns='dict'
        )

        passed_i = 0
        total_i = 0
        for t in tests:
            if t['correct']:
                passed_i += 1
            total_i += 1

        print(f'''[{i}]: acc {passed_i / total_i * 100:.2f}% ({passed_i}/{total_i})''')
        passed_count += passed_i
        total_count += total_i

    acc = f'{passed_count / (total_count) * 100:.2f}'

    write_json(save_file, {
        'acc': acc,
        'passed_count': passed_count,
        'total_count': total_count,
    })

    print(f'''\
=== {run_type} ===
acc: {acc}%
passed: {passed_count}
total: {total_count}

result save to {save_file}
''')
