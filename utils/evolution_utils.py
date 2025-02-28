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
