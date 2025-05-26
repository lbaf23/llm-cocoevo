from generators import TestGenerator
from utils import append_jsonl, read_jsonl, create_or_clear_file, init_log, create_dirs, get_codes
from tqdm import tqdm
from typing import Dict, List
from running_utils import load_env
import os
from code_evaluator import evaluate_code, get_line_cov_feedback


def GenTestsWithCov(
        index: int,
        codes: List[str],
        test_generator: TestGenerator,
        data: Dict,
        result_file: str,
        log_file: str,
        run_config: Dict,
        env_type: str,
        num_process: int,
        total_time_limit: float,
) -> None:
    """
    gen tests with cov
    """

    generations = run_config['generations']
    temperature = run_config['temperature']
    max_tokens = run_config['max_tokens']

    r = 0
    tests = []

    # load result
    result = read_jsonl(result_file)
    if len(result) > 0:
        r = result[-1]['r'] + 1
        for res in result:
            for t in res['tests']:
                tests.append(t['test'])
    else:
        create_or_clear_file(log_file)
        create_or_clear_file(result_file)

    td = tqdm(initial=r, total=generations)
    td.set_description(f'''[{index}]''')

    prompt = data['prompt']
    entry_point = data['entry_point']
    data_args = data['data_args']

    while r < generations:
        if r == 0 or len(tests) == 0:
            gen = test_generator.generate(
                prompt=prompt,
                entry_point=entry_point,
                env_type=env_type,
                data_args=data_args,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            tests = list(set(tests))
            best_code = None
            best_score = -1.0
            for code in codes:
                res = evaluate_code(
                    code=code,
                    tests=tests,
                    env_type=env_type,
                    data_args=data_args
                )
                if res['score'] > best_score:
                    best_score = res['score']
                    best_code = code

            cov = get_line_cov_feedback(
                code=best_code,
                test_cases=tests,
                env_type=env_type,
                data_args=data_args,
                num_process=num_process,
                total_time_limit=total_time_limit
            )
            program_feedback = cov['feedback']
            gen = test_generator.generate_population(
                prompt=prompt,
                entry_point=entry_point,
                env_type=env_type,
                data_args=data_args,
                generate_mode='feedback',
                max_tests_per_generation=run_config['max_tests_per_generation'],
                program_feedback=program_feedback,
                max_tokens=run_config['max_tokens'],
                temperature=run_config['temperature']
            )

        tokens = gen['tokens_count']
        all_tests = [{'test': t} for t in gen['tests']]

        for t in all_tests:
            tests.append(t['test'])

        append_jsonl(result_file, {
            'r': r,
            'tests': all_tests,
            'total_tokens_count': tokens
        })
        td.update(1)
        r += 1


if __name__ == '__main__':
    env = load_env()

    dataset = env['dataset']
    model = env['model']
    args = env['args']
    log_dir = env['log_dir']
    result_dir = env['result_dir']
    config = env['config']

    run_type = env['run_type']

    assert run_type.startswith('gen_tests_cov'), f'{run_type} is not a valid run_type'
    run_config = config[run_type]

    codes_run_type = run_config['codes']
    max_codes = run_config['max_codes']

    codes_dir = os.path.join(result_dir.rstrip(run_type), codes_run_type)
    create_dirs(result_dir)

    test_generator = TestGenerator(model)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        codes_file = os.path.join(codes_dir, f'result_{i}.jsonl')
        codes = get_codes(codes_file)
        codes = codes[ : max_codes]

        data = dataset.get_data(i)

        # create files
        log_file = os.path.join(log_dir, f'log_{i}.log')
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        init_log(log_file)

        run_config = config[args.run_type]
        GenTestsWithCov(
            index=i,
            codes=codes,
            test_generator=test_generator,
            data=data,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
            env_type=args.env_type,
            num_process=args.num_process,
            total_time_limit=args.total_time_limit,
        )
