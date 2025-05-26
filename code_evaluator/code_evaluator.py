from typing import List,Dict, Union, Any
from .py_asserts_evaluator import evaluate_py_asserts_multi_process
from .real_world_evaluator import evaluate_real_world
from .py_asserts_coverage_evaluator import get_py_asserts_line_cov_feedback
from .real_world_coverage_evaluator import get_real_world_line_cov_feedback
import os


def evaluate_code(
        code: str,
        tests: Union[str, List[str]],
        env_type: str,
        data_args: Dict[str, Any],
        num_process: int = 5,
        total_time_limit: float = 2.0,
        feedback: bool = False,
) -> Dict:
    """
    Args:
        code (str):
        tests (List[str]):
        env_type (str): 'func'
        num_process (int): number of process to run the tests
        total_time_limit (float): total time limit for all tests
        feedback (bool):
        data_args: additional arguments for the evaluator
            context (str): context of repo exec
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


    if env_type == 'func':
        res = evaluate_py_asserts_multi_process(
            code=code,
            test_cases=tests,
            total_time_limit=total_time_limit,
            num_process=num_process,
            feedback=feedback
        )
    elif env_type.startswith('real_world'):
        res = evaluate_real_world(
            code=code,
            env_type=env_type,
            data_args=data_args,
            test_cases=tests,
            num_process=num_process,
            total_time_limit=total_time_limit,
        )
    else:
        raise NotImplementedError

    return res



def get_line_cov_feedback(
        code: str,
        test_cases: List[str],
        env_type: str,
        data_args: Dict[str, Any],
        num_process: int = 5,
        total_time_limit: float = 2.0,
) -> Dict[str, Union[str, float]]:
    if type(test_cases) == str:
        test_cases = [test_cases]

    if len(test_cases) == 0:
        return {
            'score': 0.0,
            'feedbacks': [],
            'status': []
        }

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import sys
    sys.setrecursionlimit(10 ** 8)
    sys.set_int_max_str_digits(0)

    if env_type == 'func':
        return get_py_asserts_line_cov_feedback(
            code=code,
            test_cases=test_cases,
            env_type=env_type,
            data_args=data_args,
            num_process=num_process,
            total_time_limit=total_time_limit,
        )
    elif env_type == 'real_world_function':
        return get_real_world_line_cov_feedback(
            code=code,
            test_cases=test_cases,
            data_args=data_args,
            total_time_limit=total_time_limit,
        )
    else:
        raise NotImplementedError
