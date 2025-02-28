from typing import List,Dict, Union
from .py_asserts_evaluator import evaluate_py_asserts_multi_process
import os


def evaluate_code(
        code: str,
        tests: Union[str, List[str]],
        evaluator_type: str = 'func',
        num_process: int = 5,
        total_time_limit: float = 2.0,
        feedback: bool = False,
        **args
) -> Dict:
        """
        Args:
            code (str): 
            tests (List[str]):
            evaluator_type (str): 'func'
            num_process (int): number of process to run the tests
            total_time_limit (float): total time limit for all tests
        Returns:
            (Dict):
                score (float):
                feedbacks (List[Dict]): failed test case feedback message
                status (List[bool]):
        """
        if type(tests) == str:
            tests = [tests]

        if len(tests) == 0:
            return {
                'score': 0.0,
                'feedbacks': [],
                'status': []
            }

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import sys
        sys.setrecursionlimit(10 ** 8)
        sys.set_int_max_str_digits(0)

        if evaluator_type == 'func':
            res = evaluate_py_asserts_multi_process(
                code=code,
                test_cases=tests,
                total_time_limit=total_time_limit,
                num_process=num_process,
                feedback=feedback
            )
        else:
            raise NotImplementedError

        return res
