import json
from generators import CodeGenerator, TestGenerator
from code_evaluator import evaluate_code, get_line_cov_feedback
from typing import Tuple, List, Dict
from utils import read_jsonl, append_jsonl, create_or_clear_file, create_dirs, init_log, pareto_selection, make_levels
from utils.evolution_utils import selection, best_one
from tqdm import tqdm
from running_utils import load_env
import os
import random
import math


def crossover_evaluation(
        code_population: List[Dict],
        test_population: List[Dict],
        env_type: str,
        data_args: Dict,
        num_process: int,
        total_time_limit: float,
) -> Tuple[List[Dict], List[Dict], List[List[bool]]]:
    """
    evaluate code and test population
    Args:
        code_population: List[Dict]
        test_population: List[Dict]
        env_type: str
        num_process: int
        total_time_limit: float

    Returns:
        code_population: List[Dict]
        test_population: List[Dict]
        matrix: List[List[bool]]: code_population x test_population
    """

    # get strs
    test_strs = [t['test'] for t in test_population]
    code_strs = [t['code'] for t in code_population]

    code_length = len(code_population)
    test_length = len(test_population)

    matrix = [[False for _ in range(test_length)] for _ in range(code_length)]
    # calculate idx for the matrix
    for i in range(code_length):
        code_population[i]['idx'] = str(i)
    for j in range(test_length):
        test_population[j]['idx'] = str(j)

    for i, code in enumerate(code_strs):
        res = evaluate_code(
            code=code,
            tests=test_strs,
            env_type=env_type,
            data_args=data_args,
            num_process=num_process,
            total_time_limit=total_time_limit,
        )

        code_population[i]['score'] = res['score']
        code_population[i]['feedbacks'] = res['feedbacks']
        code_population[i]['status'] = res['status']

        # save matrix
        for j, passed in enumerate(res['status']):
            if passed:
                matrix[i][j] = True

    for j in range(test_length):
        score = 0.0
        for i in range(code_length):
            if matrix[i][j]:
                score += 1
        score /= code_length
        test_population[j]['score'] = score

    return code_population, test_population, matrix


def get_feedback(item: Dict) -> str:
    feedbacks = item['feedbacks']
    feedbacks = [f for f in feedbacks if f.__contains__('message')]
    if len(feedbacks) == 0:
        return ''

    return random.sample(feedbacks, 1)[0]['message']


def calculate_codet_reward_fitness(
        code_population: List[Dict],
        test_population: List[Dict],
        matrix: List[List[bool]],
        reward_func: str = 'linear'
) -> List[Dict]:
    code_nums = len(code_population)
    test_nums = len(test_population)

    for i in range(code_nums):
        assert int(code_population[i]['idx']) == i
        code_population[i]['codet_reward'] = 0.0

    test2code_dict = {}
    for i in range(code_nums):
        test_vector = tuple(sorted([j for j in range(test_nums) if matrix[i][j] == True]))
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


def calculate_test_conf(test_idx, code_population, matrix):
    passed = 0.0
    # failed = 0.0
    total = 0.0
    for c in code_population:
        code_idx = int(c['idx'])
        if matrix[code_idx][test_idx]:
            passed += c['fitness']
        total += c['fitness']

    conf = 0.0 if total == 0.0 else passed / total
    return conf


def calculate_test_disc(test_idx, code_population, matrix):
    passed = 0.0
    total = 0.0
    for c in code_population:
        code_idx = int(c['idx'])
        if matrix[code_idx][test_idx]:
            passed += 1
        total += 1
    p = 0.0 if total == 0.0 else passed / total
    return 0.0 if p == 0.0 or p == 1.0 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def cosine_annealing_scheduler(t, T, x_initial: float = 0.0, x_final: float = 1.0):
    return x_final + 0.5 * (x_initial - x_final) * (1 + math.cos(math.pi * t / T))


class CoCoEvo:
    def __init__(
            self,
            code_generator: CodeGenerator,
            test_generator: TestGenerator,
            run_config: Dict,
            env_type: str,
            num_process: int,
            total_time_limit: float,
    ) -> None:
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.run_config = run_config
        self.env_type = env_type
        self.num_process = num_process
        self.total_time_limit = total_time_limit

        self.code_config = self.run_config['code']
        self.test_config = self.run_config['test']

        self.init_config = self.run_config['init']
        self.use_random_prompt = self.init_config['use_random_prompt'] if self.init_config.__contains__('use_random_prompt') else False
        self.code_init_generations = self.init_config['code_init_generations']
        self.test_init_generations = self.init_config['test_init_generations']

        self.code_population_nums = self.code_config['population_nums']

        self.total_tokens_count = []


        self.scheduler_config = self.code_config['scheduler']

        self.iterator_rounds = int(self.code_config['max_generations'] / self.code_population_nums)
        total_gens = self.init_config['code_init_generations']

        self.code_crossover_nums = []
        self.code_mutation_nums = []

        ir = 1
        while total_gens < self.code_config['max_generations']:
            if self.scheduler_config['func'] == 'cosine':
                crossover_rate = cosine_annealing_scheduler(
                    ir,
                    self.iterator_rounds - 1,
                    self.scheduler_config['start_rate'],
                    self.scheduler_config['end_rate']
                )
            elif self.scheduler_config['func'] == 'none':
                crossover_rate = self.scheduler_config['start_rate']
            else:
                raise NotImplementedError

            crossover_nums_i = int(crossover_rate * self.code_population_nums)

            self.code_crossover_nums.append(crossover_nums_i)
            self.code_mutation_nums.append(self.code_population_nums - crossover_nums_i)

            total_gens += self.code_population_nums
            ir += 1

        self.test_calls = [self.init_config['test_init_generations']] + \
            [self.test_config['offspring']['generations'] for _ in range(self.iterator_rounds - 1)]

        assert sum(self.code_crossover_nums) + sum(self.code_mutation_nums) + self.code_init_generations == self.code_config['max_generations'], f'''code_calls: {self.code_calls}, sum: {sum(self.code_calls)}'''
        assert sum(self.test_calls) == self.test_config['max_generations'], f'''test_calls: {self.test_calls}, sum: {sum(self.test_calls)}'''

        print(f'''\
=== code ===
population_nums: {self.code_population_nums}
max_generations: {self.code_config['max_generations']}
iterator_rounds: {self.iterator_rounds}
use_random_prompt: {self.use_random_prompt}

init: {self.code_init_generations}
crossover: {self.code_crossover_nums}
mutation: {self.code_mutation_nums}

=== test ===
population_nums: {self.test_config['population_nums']}
max_generations: {self.test_config['max_generations']}
{self.test_calls}
''')


    def load_checkpoint(self):
        result = read_jsonl(self.result_file)
        if len(result) > 0:
            self.code_population = result[-1]['code_population']
            self.test_population = result[-1]['test_population']
            self.matrix = result[-1]['matrix']
            self.r = result[-1]['r'] + 1
            self.code_generations = result[-1]['code_generations']
        else:
            create_or_clear_file(self.result_file)
            create_or_clear_file(self.log_file)

    def should_return(self):
        return self.code_generations >= self.code_config['max_generations']

    def update(self, tokens_count: Dict[str, int]) -> None:
        """
        Update tqdm, code generations, and tokens count
        """
        self.code_generations += 1
        self.td.update(1)
        self.total_tokens_count.append(tokens_count)

    def update_code_fitness(self, fitness_function: str = '') -> None:
        if fitness_function == '':
            fitness_function = self.code_config['fitness_function']

        if fitness_function == 'score':
            for c in self.code_population:
                c['fitness'] = c['score']
        elif fitness_function.startswith('codet'):
            reward_func = 'linear'
            if fitness_function.__contains__('_'):
                reward_func = fitness_function.split('_')[1]
            self.code_population = calculate_codet_reward_fitness(
                self.code_population,
                self.test_population,
                self.matrix,
                reward_func=reward_func
            )
            for c in self.code_population:
                c['fitness'] = c['codet_reward']
        else:
            raise NotImplementedError(f'''no such code fitness function: {fitness_function}''')

    def update_test_fitness(self, fitness_function: str = '') -> None:
        if fitness_function == '':
            fitness_function = self.test_config['fitness_function']

        if fitness_function == 'score':
            for t in self.test_population:
                t['fitness'] = t['score']
        elif fitness_function == 'weighted':
            for t in self.test_population:
                t['confidence'] = calculate_test_conf(int(t['idx']), self.code_population, self.matrix)
                t['fitness'] = t['confidence']
        elif fitness_function == 'mix':
            for t in self.test_population:
                t['discrimination'] = calculate_test_disc(int(t['idx']), self.code_population, self.matrix)
                t['confidence'] = calculate_test_conf(int(t['idx']), self.code_population, self.matrix)
                t['fitness'] = t['confidence']
        elif fitness_function == 'failed_rate':
            for t in self.test_population:
                t['failed_rate'] = 1 - t['score']
                t['fitness'] = t['failed_rate']
        else:
            raise NotImplementedError(f'''no such test fitness function: {fitness_function}''')

    def select_tests(self, selection_algo: str = ''):
        self.test_offspring = self.test_population

        selection_size = self.test_config['population_nums']
        if type(selection_size) == str:
            if selection_size.startswith('auto'):
                extra_nums = int(selection_size.split('_')[1])
                passed_all_nums = 0
                for t in self.test_population:
                    if t['score'] == 1.0:  # test passed all the codes
                        passed_all_nums += 1
                selection_size = max(10, passed_all_nums + extra_nums)
            else:
                raise NotImplementedError(f'no such test selection_size: {selection_size}')

        if selection_algo == '':
            selection_algo = self.test_config['selection']['selection_algo']

        if selection_algo == 'pareto':
            filter_algo = ''
            if self.test_config['selection'].__contains__('filter_algo'):
                filter_algo = self.test_config['selection']['filter_algo']

            self.test_population = pareto_selection(
                self.test_population,
                selection_size,
                metrics=['confidence', 'discrimination'],
                filter_algo=filter_algo
            )
        elif selection_algo == 'avg':
            avg_fitness = 0
            for t in self.test_population:
                avg_fitness += t['fitness']
            avg_fitness /= len(self.test_population)
            self.test_population = [t for t in self.test_population if t['fitness'] >= avg_fitness]
        elif selection_algo == 'levels':
            levels = make_levels(self.test_population, 'fitness')
            selected = []
            for level in levels:
                selected += level
                if len(selected) >= selection_size:
                    break
            self.test_population = selected
        elif selection_algo == '1st':
            max_confidence = 0.0
            for t in self.test_population:
                max_confidence = max(max_confidence, t['confidence'])

            selected = [t for t in self.test_population if t['confidence'] == max_confidence]
            left = [t for t in self.test_population if t['confidence'] < max_confidence]
            if len(selected) < selection_size:
                left = sorted(left, key=lambda i:i['confidence'], reverse=True)
                selected += left[ : selection_size - len(selected)]
            self.test_population = selected
        else:
            selected_index = selection(
                population=self.test_population,
                select_size=selection_size,
                algo=selection_algo,
                metric='fitness'
            )
            self.test_population = [self.test_population[i] for i in selected_index]

    def select_codes(self, selection_algo: str = ''):
        if selection_algo == '':
            selection_algo = self.code_config['selection']['selection_algo']

        select_size = self.code_config['population_nums']
        selected_index = selection(
            population=self.code_population,
            select_size=select_size,
            algo=selection_algo,
            metric='fitness'
        )
        self.code_population = [self.code_population[i] for i in selected_index]

    def do_init(self):
        self.code_population = []
        self.test_population = []

        # init_method = self.init_config['method']
        code_init_generations = self.init_config['code_init_generations']
        test_init_generations = self.init_config['test_init_generations']

        # init code population
        self.code_offspring = []
        for _ in range(code_init_generations):
            if self.should_return():
                break

            gen = code_generator.generate(
                prompt=self.prompt,
                env_type=self.env_type,
                data_args=self.data_args,
                init_method='random_prompt' if self.use_random_prompt else 'default',
                max_tokens=self.code_config['max_tokens'],
                temperature=self.code_config['temperature']
            )
            self.update(gen['tokens_count'])

            item = {
                'r': self.r,
                'stage': 'init',
                'code': gen['code'],
                'repair_rounds': 0,
                'repaired_nums': 0
            }
            self.code_offspring.append(item)

        self.update_code_population()

        self.test_offspring = []
        for _ in range(test_init_generations):
            if len(self.test_offspring) == 0:
                gen = test_generator.generate_population(
                    prompt=self.prompt,
                    entry_point=self.entry_point,
                    env_type=self.env_type,
                    data_args=self.data_args,
                    generate_mode='random',
                    max_tests_per_generation=self.test_config['max_tests_per_generation'],
                    max_tokens=self.test_config['max_tokens'],
                    temperature=self.test_config['temperature']
                )
            else:
                gen = test_generator.generate_population(
                    prompt=self.prompt,
                    entry_point=self.entry_point,
                    env_type=self.env_type,
                    data_args=self.data_args,
                    generate_mode='population',
                    existing_tests=self.test_offspring,
                    max_tests_per_generation=self.test_config['max_tests_per_generation'],
                    max_feedback_tests=self.test_config['offspring']['max_feedback_tests'],
                    max_tokens=self.test_config['max_tokens'],
                    temperature=self.test_config['temperature']
                )
            self.test_offspring += gen['tests']

            # count tokens
            self.total_tokens_count.append(gen['tokens_count'])

        self.update_test_population()
        self.update_fitness(update_codes=True, update_tests=False)

    def code_crossover(self):
        crossover_nums = self.code_crossover_nums[self.r - 1]

        self.code_crossover_offspring = []
        for _ in range(crossover_nums):
            if self.should_return():
                break

            parents_index = selection(
                population=self.code_population,
                select_size=2,
                algo=self.code_config['crossover']['selection_algo'],
                metric='fitness'
            )
            parent1 = self.code_population[parents_index[0]]
            parent2 = self.code_population[parents_index[1]]

            gen = code_generator.generate_crossover(
                prompt=self.prompt,
                code1=parent1['code'],
                code2=parent2['code'],
                env_type=self.env_type,
                data_args=self.data_args,
                max_tokens=self.code_config['max_tokens'],
                temperature=self.code_config['temperature']
            )
            self.update(gen['tokens_count'])

            item = {
                'r': self.r,
                'code': gen['code'],
                'stage': 'crossover',
                'output': gen['output'],
                'repair_rounds': 0,
                'repaired_nums': 0,
                'parents_index': parents_index,
            }
            self.code_crossover_offspring.append(item)

    def code_mutation(self):
        mutation_nums = self.code_mutation_nums[self.r - 1]

        self.code_mutation_offspring = []
        parents_index = selection(
            population=self.code_population,
            select_size=mutation_nums,
            algo=self.code_config['mutation']['selection_algo'],
            metric='fitness'
        )
        for p_index in parents_index:
            if self.should_return():
                break

            parent = self.code_population[p_index]
            gen = code_generator.generate_mutation(
                prompt=self.prompt,
                code=parent['code'],
                env_type=self.env_type,
                data_args=self.data_args,
                max_tokens=self.code_config['max_tokens'],
                temperature=self.code_config['temperature']
            )
            self.update(gen['tokens_count'])
            item = {
                'r': self.r,
                'code': gen['code'],
                'stage': 'mutation',
                'repair_rounds': 0,
                'repaired_nums': 0,
                'parent_index': p_index,
            }
            self.code_mutation_offspring.append(item)

    def update_code_population(self):
        self.code_population += self.code_offspring

    def generate_test_offspring(self) -> None:
        self.test_offspring = []

        offspring_config = self.test_config['offspring']
        generations = offspring_config['generations']

        # make test feedback
        existing_tests = [t['test'] for t in self.test_population]
        program_feedback = ''
        if offspring_config['method'] == 'population_and_feedback':
            feedback_code = best_one(self.code_population, metric='fitness')['code']
            cov = get_line_cov_feedback(
                code=feedback_code,
                test_cases=existing_tests,
                env_type=self.env_type,
                data_args=self.data_args,
                num_process=self.num_process,
                total_time_limit=self.total_time_limit
            )
            program_feedback = cov['feedback']
        for _ in range(generations):
            # generate test offspring
            gen = test_generator.generate_population(
                prompt=self.prompt,
                entry_point=self.entry_point,
                env_type=self.env_type,
                data_args=self.data_args,
                existing_tests=existing_tests,
                generate_mode=offspring_config['method'],
                max_tests_per_generation=self.test_config['max_tests_per_generation'],
                max_feedback_tests=offspring_config['max_feedback_tests'],
                program_feedback=program_feedback,
                max_tokens=self.test_config['max_tokens'],
                temperature=self.test_config['temperature']
            )
            self.test_offspring += gen['tests']

            # count tokens
            self.total_tokens_count.append(gen['tokens_count'])

    def update_test_population(self) -> None:
        tests_set = set()
        for p in self.test_population:
            tests_set.add(p['test'])

        for test in self.test_offspring:
            if not tests_set.__contains__(test):
                item = {
                    'r': self.r,
                    'test': test
                }
                self.test_population.append(item)
                tests_set.add(test)

    def update_fitness(
            self,
            code_fitness_function: str = '',
            test_fitness_function: str = '',
            update_codes: bool = True,
            update_tests: bool = True
    ) -> None:
        """
        crossover_evaluation --> update fitness

        """
        self.code_population, self.test_population, self.matrix = crossover_evaluation(
            self.code_population,
            self.test_population,
            self.env_type,
            self.data_args,
            self.num_process,
            self.total_time_limit
        )
        if update_codes:
            self.update_code_fitness(code_fitness_function)
        if update_tests:
            self.update_test_fitness(test_fitness_function)

    def save_result(self):
        """
        Save all results after one iteration
        """
        append_jsonl(result_file, {
            'r': self.r,
            'code_population': self.code_population,
            'test_population': self.test_population,
            'code_offspring': self.code_offspring,
            'test_offspring': self.test_offspring,
            'matrix': self.matrix,
            'code_generations': self.code_generations,
            'total_tokens_count': self.total_tokens_count
        })
        self.total_tokens_count = []

    def run(
            self,
            index: int,
            data: Dict,
            result_file: str,
            log_file: str
    ) -> None:
        self.index = index
        self.result_file = result_file
        self.log_file = log_file

        self.prompt = data['prompt']
        self.entry_point = data['entry_point']
        self.data_args = data['data_args']

        self.r = 0
        self.code_generations = 0
        self.code_population = []
        self.test_population = []

        self.test_offspring = []
        self.code_offspring = []

        self.load_checkpoint()

        self.td = tqdm(initial=self.code_generations, total=self.code_config['max_generations'])
        self.td.set_description(f'''[{index}]''')


        while not self.should_return():
            if self.r == 0:
                self.do_init()
            else:
                ###### generate code offspring ######
                self.code_crossover_offspring, self.code_mutation_offspring, self.code_repair_offspring = [], [], []
                if not self.should_return():
                    self.code_crossover()

                if not self.should_return():
                    self.code_mutation()

                self.code_offspring = self.code_crossover_offspring + self.code_mutation_offspring + self.code_repair_offspring
                self.update_code_population()

                # evaluate codes on tests
                self.update_fitness(update_codes=True, update_tests=False)
                self.select_codes()


                ###### generate test offspring ######
                self.generate_test_offspring()
                self.update_test_population()

                # evaluate tests on codes
                self.update_fitness(update_codes=False, update_tests=True)
                self.select_tests()

            self.update_fitness()

            # best = best_one(self.code_population, metric='fitness')

            self.save_result()
            self.r += 1


if __name__ == '__main__':
    env = load_env()

    dataset = env['dataset']
    model = env['model']
    args = env['args']
    log_dir = env['log_dir']
    result_dir = env['result_dir']
    config = env['config']
    run_type = env['run_type']

    assert run_type.startswith('coevod'), f'run_type {run_type} does not start with coevod'

    create_dirs(result_dir)
    create_dirs(log_dir)

    run_config = config[run_type]

    code_generator = CodeGenerator(model)
    test_generator = TestGenerator(model)

    print('=== config ===', flush=True)
    print(json.dumps(run_config, indent=4), flush=True)

    evo = CoCoEvo(
        code_generator=code_generator,
        test_generator=test_generator,
        run_config=run_config,
        env_type=args.env_type,
        num_process=args.num_process,
        total_time_limit=args.total_time_limit,
    )

    for i in dataset.data_range:
        if args.start < args.end and not args.start <= i < args.end:
            continue

        data = dataset.get_data(i)

        # create files
        result_file = os.path.join(result_dir, f'result_{i}.jsonl')
        log_file = os.path.join(log_dir, f'log_{i}.log')

        init_log(log_file)

        evo.run(
            index=i,
            data=data,
            result_file=result_file,
            log_file=log_file,
        )
