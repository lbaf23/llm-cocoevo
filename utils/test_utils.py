from typing import Dict, List, Union
from .jsonl_utils import read_jsonl


def get_codes(file_path: str) -> List[str]:
    content = read_jsonl(file_path)
    codes = []
    for c in content:
        codes.append(c['code'])
    return codes


def get_unique_tests(
        file_path: str,
        max_tests_generations: int = 5,
        max_tests_per_generation: int = 10,
        returns: str = 'str'
) -> Union[List[str] | List[Dict]]:
    tests = []
    tests_set = set()

    content = read_jsonl(file_path)
    content = content[ : max_tests_generations]
    for c in content:
        tests_i = c['tests'][ : max_tests_per_generation]
        for t in tests_i:
            if not tests_set.__contains__(t['test']):
                tests_set.add(t['test'])

                if returns == 'str':
                    tests.append(t['test'])
                elif returns == 'dict':
                    tests.append(t)
                else:
                    raise ValueError('Invalid return type')
    return tests


def extract_test_inputs(output: str) -> List[str]:
    output = output.strip()
    if not output.endswith('```'):
        lines = output.split('\n')[ : -1]
    else:
        lines = output.split('\n')

    test_inputs = []
    for l in lines:
        if l.startswith('input:'):
            test_inputs.append(l[l.index('input:') + len('input:') : ].strip())
    return test_inputs
