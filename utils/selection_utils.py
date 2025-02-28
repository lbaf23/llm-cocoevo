from typing import List, Dict
import Levenshtein
import math


def calculate_distance(items: List[Dict], key: str, target: str = 'distance') -> List[Dict]:
    """
    D_i = \sum_{j \neq i} d_{ij} / (n - 1)
    """
    n = len(items)
    for i in range(n):
        d = 0
        for j in range(n):
            if i != j:
                d += Levenshtein.distance(items[i][key], items[j][key])
        items[i][target] = d
    for i in range(n):
        items[i][target] /= n - 1
    return items


def calculate_crowding_distance(items: List[Dict], metrics: List[str]) -> List[float]:
    """
    calculate crowding distance
    """
    n = len(items)
    m = len(metrics)

    for i in range(n):
        items[i]['crowding_distance'] = 0

    for i in range(m):
        metric = metrics[i]
        cd = [0 for _ in range(n)]

        # sorted by m-th metric
        items = sorted(items, key=lambda x: x[metric])
        fmax = items[-1][metric]
        fmin = items[0][metric]

        # boundary points have infinite distance
        cd[0] = math.inf
        cd[n - 1] = math.inf

        # calculate crowding distance for intermediate points
        for i in range(1, n - 1):
            if fmax - fmin == 0:
                d = 0
            else:
                d = (
                    (items[i + 1][metric] - items[i - 1][metric]) /
                    (fmax - fmin)
                )
            cd[i] = d

        for i in range(n):
            items[i]['crowding_distance'] += cd[i]

    for i in range(n):
        items[i]['crowding_distance'] /= len(metrics)

    return items


def build_pareto_front(test_population: List[Dict], metrics: List[str]) -> List[List[Dict]]:
    def is_dominated(item1: Dict, item2: Dict, metrics: List[str]) -> bool:
        """
        return True if item2 dominates item1, otherwise False
        """
        return all(item2[m] >= item1[m] for m in metrics) and \
                any(item2[m] > item1[m] for m in metrics)

    pareto_front = [[] for _ in range(len(test_population))]
    for i, item1 in enumerate(test_population):
        dominated = 0
        for j, item2 in enumerate(test_population):
            if i != j and is_dominated(item1, item2, metrics):
                dominated += 1

        # dominated by ... items
        pareto_front[dominated].append(item1)
    return pareto_front


def pareto_selection(test_population: List[Dict], select_size: int, metrics: List[str], mode: str = 'auto', filter_algo: str = '') -> List[Dict]:

    pareto_front = build_pareto_front(test_population, metrics)
    selected = pareto_front[0]

    for i in range(1, len(pareto_front)):
        if len(selected) >= select_size:
            break

        p = pareto_front[i]

        if len(selected) + len(p) <= select_size:
            selected += p
        else:
            if mode == 'strict':
                p = sorted(p, key=lambda x: x['fitness'], reverse=True)
                p = p[: select_size - len(selected)]
                selected += p
            elif mode == 'auto':
                selected += p
            else:
                raise NotImplementedError
    
    if filter_algo == 'avg':
        avg_fitness = 0.0
        for t in test_population:
            avg_fitness += t['fitness']
        avg_fitness /= len(test_population)

        selected = [s for s in selected if s['fitness'] >= avg_fitness]

    return selected


def make_levels(test_population: List[Dict], metric: str = 'fitness') -> List[List[Dict]]:
    levels = []
    test_population = sorted(test_population, key=lambda i: i['fitness'], reverse=True)
    item = None
    level = []
    for t in test_population:
        if item == None:
            item = t
            level = [t]
        elif t[metric] == item[metric]:
            level.append(t)
        else:
            levels.append(level)
            level = [t]
            item = t

    if len(level) > 0:
        levels.append(level)
    return levels
