from typing import List, Dict, Tuple
import random
from Levenshtein import distance


def sort_population(population: List[Dict]) -> List[Dict]:
    return sorted(population, key=lambda x: x['score'], reverse=True)


def best_one(items: List[Dict], metric: str = 'fitness') -> Dict:
    assert len(items) > 0

    best = items[0]
    for p in items:
        if p[metric] > best[metric]:
            best = p
    return best


def greedy_selection(
        items: List[Dict],
        select_size: int,
        metric: str
) -> List[int]:
    index_list = [i for i in range(len(items))]
    random.shuffle(index_list)
    sorted_index_list = sorted(index_list, key=lambda i: items[i][metric], reverse=True)
    return sorted_index_list[ : select_size]


def tournament_selection(
        population: List[Dict],
        select_size: int,
        metric: str,
        put_back: bool = False,
        k: int = 3,
) -> List[int]:
    index_selected_list = []
    index_list = [i for i in range(len(population))]
    for _ in range(select_size):
        assert k <= len(index_list)
        index = random.sample(index_list, k)
        random.shuffle(index)

        winner_index = index[0]
        for i in index:
            if population[i][metric] > population[winner_index][metric]:
                winner_index = i

        index_selected_list.append(winner_index)
        # non-replacement
        if not put_back:
            index_list.remove(winner_index)
    return index_selected_list


def greedy_random_selection(
        population: List[Dict],
        select_size: int,
        metric: str,
        greedy_ratio: float = 0.5
) -> List[int]:
    greedy_selection_nums = int(select_size * greedy_ratio)
    random_selection_nums = select_size - greedy_selection_nums

    index_list = [i for i in range(len(population))]
    random.shuffle(index_list)
    sorted_index_list = sorted(index_list, key=lambda i: population[i][metric], reverse=True)

    # greedy
    greedy_selected_index = sorted_index_list[ : greedy_selection_nums]
    # random
    left_index_list = sorted_index_list[greedy_selection_nums : ]
    random_selected_index = random.sample(left_index_list, random_selection_nums)

    return greedy_selected_index + random_selected_index


def selection(
        population: List[Dict],
        select_size: int,
        algo: str,
        metric: str = 'fitness',
        **args
) -> List[int]:
    """
    Args:
        population (List[Dict]):
        select_size (int):
        algo (str): greedy, tournament_k, random
        metric (str): score, distance

    Returns:
        List[int]: selected index list
    """
    if len(population) <= select_size:
        return [i for i in range(len(population))]
    if select_size == 0:
        return []

    assert len(population) > select_size

    if algo == 'greedy':
        return greedy_selection(population, select_size, metric)
    elif algo.startswith('tournament'):
        if args.__contains__('k'):
            k = int(args['k'])
            args.pop('k')
        elif algo.__contains__('_'):
            k = int(algo.split('_')[1])
        else:
            k = 3
        return tournament_selection(population, select_size, metric, k=k, **args)
    elif algo == 'random':
        return random_selection(population, select_size)
    elif algo.startswith('greedy_random'):
        if args.__contains__('greedy_ratio'):
            greedy_ratio = float(args['greedy_ratio'])
            args.pop('greedy_ratio')
        elif algo.count('_') > 1:
            greedy_ratio = float(algo.split('_')[2])
        else:
            greedy_ratio = 0.5
        return greedy_random_selection(population, select_size, metric, greedy_ratio=greedy_ratio)
    elif algo == 'test_distance':
        assert select_size == 2
        return test_distance_selection(population, **args)
    else:
        raise NotImplementedError


def survival_selection(
        population: List[Dict],
        select_size: int,
        algo: str = 'greedy',
) -> List[Dict]:
    assert len(population) >= select_size

    selected_index = selection(population, select_size, algo)
    return [population[i] for i in selected_index]


def matting_selection(
        population: List[Dict],
        select_size: int,
        algo: str = 'random',
        **args
) -> List[int]:
    assert len(population) >= select_size

    if algo == 'none':
        return [i for i in range(len(population))]
    else:
        return selection(population, select_size, algo, **args)


def random_selection(
        population: List[Dict],
        select_size: int
) -> List[int]:
    assert len(population) >= select_size
    index_list = [i for i in range(len(population))]
    return random.sample(index_list, select_size)


def test_distance_selection(
        population: List[Dict],
        **args
) -> List[int]:
    length = len(population)

    index_list = [i for i in range(length)]
    i1 = random.sample(index_list, 1)[0]
    i2 = 0
    max_d = -1
    for i in range(length):
        if i == i1:
            continue
        d = distance(population[i]['status'], population[i1]['status'])
        if d > max_d:
            i2 = i
            max_d = d
    return [i1, i2]


def calculate_distance(codes: List[str]) -> Tuple[float, List[List[float]]]:
    n = len(codes)
    distance_map = [[0 for _ in range(n)] for _ in range(n)]
    total_distance = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance_map[i][j] = distance(codes[i], codes[j])
            total_distance += distance_map[i][j]
            count += 1
    for i in range(n):
        for j in range(i):
            distance_map[i][j] = distance_map[j][i]

    avg_distance = total_distance / count if count > 0 else total_distance
    return avg_distance, distance_map


def shared_fitness(
        population: List[Dict],
        sigma_share: float = 0.0,
        alpha: int = 1
) -> List[float]:
    """
    calculate the shared fitness
    """
    codes = [p['instance'] for p in population]
    scores = [p['score'] for p in population]

    avg_distance, distance_map = calculate_distance(codes)
    if sigma_share == 0:
        factor = 0.3
        sigma_share = avg_distance * factor

    fitness_list = []
    n = len(codes)
    for i in range(n):
        s_count = 0
        for j in range(n):
            d = distance_map[i][j]
            sh = max(0, 1 - (d / sigma_share) ** alpha) if d < sigma_share else 0
            s_count += sh

        f = scores[i] / s_count if s_count > 0 else scores[i]
        fitness_list.append(f)
    return fitness_list


def fitness(population: List[Dict], fitness_algo: str, **args) -> List[float]:
    if fitness_algo == '' or fitness_algo == 'score_fitness':
        return [p['score'] for p in population]
    elif fitness_algo == 'shared_fitness':
        return shared_fitness(population, **args)
    else:
        raise NotImplementedError


def worst_one_index(items: List[Dict]) -> int:
    assert len(items) > 0

    worst_index = 0
    for i, p in enumerate(items):
        if p['score'] < items[worst_index]['score']:
            worst_index = i
    return worst_index
