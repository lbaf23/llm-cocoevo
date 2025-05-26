import json
from generators import CodeGenerator
from code_evaluator import evaluate_code
from typing import Tuple, List, Dict
from utils import read_jsonl, append_jsonl, create_or_clear_file, create_dirs, init_log
from utils.evolution_utils import selection
from tqdm import tqdm
from running_utils import load_env
import os
from utils import get_unique_tests
import math


def calculate_codet_reward_fitness(
        code_population: List[Dict],
        reward_func: str = 'linear'
) -> List[Dict]:
    code_nums = len(code_population)
    test_nums = len(code_population[0]['status'])

    for i in range(code_nums):
        code_population[i]['codet_reward'] = 0.0

    test2code_dict = {}
    for i in range(code_nums):
        test_vector = tuple([j for j in range(test_nums) if code_population[i]['status'][j] == True])
        if test2code_dict.__contains__(test_vector):
            test2code_dict[test_vector].append(i)
        else:
            test2code_dict[test_vector] = [i]

    for test_vector in test2code_dict.keys():
        code_vector = test2code_dict[test_vector]
        if reward_func == 'linear':
            codet_reward = len(code_vector) * len(test_vector)
        elif reward_func == 'sqrt':
            codet_reward = math.sqrt(len(code_vector)) * len(test_vector)
        else:
            raise NotImplementedError

        codet_reward = 0.0 if (code_nums * test_nums) == 0 else codet_reward / (code_nums * test_nums)
        for i in code_vector:
            code_population[i]['codet_reward'] = codet_reward

    return code_population


def update_code_fitness(
        code_population: List[Dict],
        fitness_function: str,
) -> None:
    if fitness_function.startswith('codet'):
        reward_func = 'linear'
        if fitness_function.__contains__('_'):
            reward_func = fitness_function.split('_')[1]

        code_population = calculate_codet_reward_fitness(
            code_population,
            reward_func=reward_func
        )
        for c in code_population:
            c['fitness'] = c['codet_reward']
    elif fitness_function == 'score':
        for c in code_population:
            c['fitness'] = c['score']
    else:
        raise NotImplementedError(f'fitness function {fitness_function} not implemented')


def cosine_annealing_scheduler(t, T, x_initial: float = 0.0, x_final: float = 1.0):
    return x_final + 0.5 * (x_initial - x_final) * (1 + math.cos(math.pi * t / T))


def Evolution(
        index: int,
        code_generator: CodeGenerator,
        tests: List[str],
        data: Dict,
        result_file: str,
        log_file: str,
        run_config: Dict,
        env_type: str,
        num_process: int,
        total_time_limit: float,
) -> None:
    code_population_nums = run_config['population_nums']
    max_generations = run_config['max_generations']

    crossover_config = run_config['crossover']
    mutation_config = run_config['mutation']
    scheduler_config = run_config['scheduler']

    iterator_rounds = int(max_generations / code_population_nums)


    code_crossover_nums = []
    code_mutation_nums = []

    total_gens = code_population_nums
    ir = 1
    while total_gens < max_generations:
        if scheduler_config['func'] == 'cosine':
            crossover_rate = cosine_annealing_scheduler(
                ir,
                iterator_rounds - 1,
                scheduler_config['start_rate'],
                scheduler_config['end_rate']
            )
        else:
            raise NotImplementedError

        crossover_nums_i = int(crossover_rate * code_population_nums)

        code_crossover_nums.append(crossover_nums_i)
        code_mutation_nums.append(code_population_nums - crossover_nums_i)

        total_gens += code_population_nums
        ir += 1


    prompt = data['prompt']

    r = 0
    code_generations = 0
    code_population = []
    code_crossover_offspring = []
    code_mutation_offspring = []

    result = read_jsonl(result_file)
    if len(result) > 0:
        code_population = result[-1]['code_population']
        r = result[-1]['r'] + 1
        code_generations = result[-1]['code_generations']
    else:
        create_or_clear_file(result_file)
        create_or_clear_file(log_file)

    td = tqdm(initial=code_generations, total=max_generations)
    td.set_description(f'''[{index}]''')

    while code_generations < max_generations:
        total_tokens_count = []

        if r == 0:
            # init code population
            code_population = []
            code_offspring = []
            for _ in range(code_population_nums):
                if code_generations >= max_generations:
                    break

                gen = code_generator.generate(
                    prompt=prompt,
                    env_type=env_type,
                    data_args=data['data_args'],
                    init_method=run_config['init_method'],
                    max_tokens=run_config['max_tokens'],
                    temperature=run_config['temperature']
                )
                total_tokens_count.append(gen['tokens_count'])
                code_generations += 1
                td.update(1)
                code = gen['code']
                res = evaluate_code(
                    code=code,
                    tests=tests,
                    env_type=env_type,
                    data_args=data['data_args'],
                    num_process=num_process,
                    total_time_limit=total_time_limit
                )

                item = {
                    'r': r,
                    'stage': 'init',
                    'code': gen['code'],
                    'score': res['score'],
                    'status': res['status'],
                    'feedbacks': res['feedbacks'],
                }
                code_offspring.append(item)
            update_code_fitness(code_offspring, run_config['fitness_function'])
        else:
            crossover_nums = code_crossover_nums[r - 1]
            mutation_nums = code_mutation_nums[r - 1]

            # do crossover
            code_crossover_offspring = []
            for _ in range(crossover_nums):
                if code_generations >= max_generations:
                    break

                parents_index = selection(
                    code_population,
                    2,
                    algo=crossover_config['selection_algo']
                )
                parent1 = code_population[parents_index[0]]
                parent2 = code_population[parents_index[1]]

                gen = code_generator.generate_crossover(
                    prompt=prompt,
                    code1=parent1['code'],
                    code2=parent2['code'],
                    env_type=env_type,
                    data_args=data['data_args'],
                    max_tokens=run_config['max_tokens'],
                    temperature=run_config['temperature']
                )
                total_tokens_count.append(gen['tokens_count'])
                code_generations += 1
                td.update(1)
                code = gen['code']
                res = evaluate_code(
                    code=code,
                    tests=tests,
                    env_type=env_type,
                    data_args=data['data_args'],
                    num_process=num_process,
                    total_time_limit=total_time_limit
                )

                item = {
                    'r': r,
                    'code': gen['code'],
                    'stage': 'crossover',
                    'score': res['score'],
                    'parents_index': parents_index,
                    'status': res['status'],
                    'feedbacks': res['feedbacks'],
                    'output': gen['output']
                }
                code_crossover_offspring.append(item)

            # do mutation
            code_mutation_offspring = []
            parents_index = selection(
                population=code_population,
                select_size=mutation_nums,
                algo=mutation_config['selection_algo'],
                metric='fitness'
            )
            for p_index in parents_index:
                if code_generations >= max_generations:
                    break

                parent = code_population[p_index]
                gen = code_generator.generate_mutation(
                    prompt=prompt,
                    code=parent['code'],
                    env_type=env_type,
                    data_args=data['data_args'],
                    max_tokens=run_config['max_tokens'],
                    temperature=run_config['temperature']
                )
                total_tokens_count.append(gen['tokens_count'])
                code_generations += 1
                td.update(1)
                code = gen['code']
                res = evaluate_code(
                    code=code,
                    tests=tests,
                    env_type=env_type,
                    data_args=data['data_args'],
                    num_process=num_process,
                    total_time_limit=total_time_limit
                )
                item = {
                    'r': r,
                    'code': code,
                    'score': res['score'],
                    'stage': 'mutation',
                    'parent_index': p_index,
                    'status': res['status'],
                    'feedbacks': res['feedbacks'],
                    'output': gen['output']
                }
                code_mutation_offspring.append(item)

            # update code population
            code_offspring = code_crossover_offspring + code_mutation_offspring

            update_code_fitness(code_offspring, run_config['fitness_function'])

        code_population += code_offspring
        selected_index = selection(
            population=code_population,
            select_size=code_population_nums,
            algo=run_config['selection']['selection_algo'],
            metric='fitness'
        )
        code_population = [code_population[i] for i in selected_index]

        # save result
        append_jsonl(result_file, {
            'r': r,
            'code_population': code_population,
            'code_offspring': code_offspring,
            'code_generations': code_generations,
            'total_tokens_count': total_tokens_count
        })
        r += 1

        if code_generations >= max_generations:
            break


if __name__ == '__main__':
    env = load_env()

    dataset = env['dataset']
    model = env['model']
    args = env['args']
    log_dir = env['log_dir']
    result_dir = env['result_dir']
    config = env['config']
    run_type = env['run_type']

    assert run_type.startswith('evolutiond'), 'run_type should be startswith evolution'

    run_config = config[run_type]

    create_dirs(result_dir)
    create_dirs(log_dir)

    test_config = run_config['test']

    tests_dir = os.path.join(result_dir.rstrip(run_type), test_config['tests_dir'])

    max_tests_generations = test_config['max_generations']
    max_tests_per_generation = test_config['max_tests_per_generation']

    code_generator = CodeGenerator(model)

    print('=== config ===', flush=True)
    print(json.dumps(run_config, indent=4), flush=True)

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset.get_data(i)

        tests_file = os.path.join(tests_dir, f'result_{i}.jsonl')
        tests = get_unique_tests(tests_file, max_tests_generations, max_tests_per_generation)

        # create files
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        log_file = os.path.join(log_dir, f'log_{i}.log')

        init_log(log_file)

        Evolution(
            index=i,
            code_generator=code_generator,
            tests=tests,
            data=data,
            result_file=result_file,
            log_file=log_file,
            run_config=run_config,
            env_type=args.env_type,
            num_process=args.num_process,
            total_time_limit=args.total_time_limit,
        )
