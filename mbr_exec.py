"""
MBR-Exec

"""

from code_evaluator import evaluate_code
from typing import List
from utils import write_jsonl, get_unique_tests, get_codes, create_dirs
from running_utils import load_env
from tqdm import tqdm
import os


def MBR_Exec(
        index: int,
        codes: List[str],
        tests: List[str],
        result_file: str,
        env_type: str,
        num_process: int,
        total_time_limit: int
) -> None:
    code_nums = len(codes)
    test_nums = len(tests)

    program_output_list = [['' for _ in range(test_nums)] for _ in range(code_nums)]

    for i, code in enumerate(tqdm(codes, desc=f'[{index}]')):
        res = evaluate_code(
            code=code,
            tests=tests,
            evaluator_type=env_type,
            num_process=num_process,
            total_time_limit=total_time_limit,
            feedback=True
        )
        for j in range(test_nums):
            if res['feedbacks'][j].__contains__('program_output'):
                if res['feedbacks'][j]['program_output'] == 'timeout error' or res['feedbacks'][j]['program_output'] == '':
                    # failed to execute on the test case, no program output collected
                    program_output_list[i][j] = None
                else:
                    # failed the test case, but can get the program output
                    program_output_list[i][j] = res['feedbacks'][j]['program_output']
            else:
                # execute successfully on the test input
                program_output_list[i][j] = 'truth output'

    def loss_p(program_output1, program_output2) -> int:
        for i in range(len(program_output1)):
            if program_output1[i] == None or \
                program_output2[i] == None or \
                program_output1[i] != program_output2[i]:
                return 1
        return 0

    res = []
    for i in range(code_nums):
        mbr_loss = 0
        for i2 in range(code_nums):
            if i != i2:
                mbr_loss += loss_p(program_output_list[i], program_output_list[i2])
        
        res.append({
            'i': i,
            'code': codes[i],
            'mbr_loss': mbr_loss
        })

    res = sorted(res, key=lambda x: x['mbr_loss'])
    write_jsonl(result_file, res)


if __name__ == '__main__':
    env = load_env(load_models=False)

    args = env['args']
    config = env['config']
    run_type = env['run_type']
    dataset = env['dataset']
    
    assert run_type.startswith('mbr_exec')

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

        MBR_Exec(
            index=i,
            codes=codes,
            tests=tests,
            result_file=result_file,
            env_type=args.env_type,
            num_process=args.num_process,
            total_time_limit=args.total_time_limit
        )
