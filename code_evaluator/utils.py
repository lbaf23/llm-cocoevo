import ast
from multiprocessing import Process, Queue
from typing import Tuple, Any, Callable, Dict


def runner(target: Callable, args: Tuple, q: Queue):
    res = target(*args)
    if res is not None:
        q.put(res)


def run_with_time_limit(target: Callable, args: Tuple, time_limit: float = 1.0) -> Tuple[bool, Any]:
    """
    Args:
        target: target function to be called
    Returns:
        succeed: bool
        result: Any
    """
    q = Queue()
    p = Process(target=runner, args=(target, args, q))
    p.start()
    p.join(time_limit)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, None
    try:
        return True, q.get(block=False)
    except Exception:
        return False, None


def execute(code: str, test: str) -> Dict:
    execs = []
    try:
        execs.append(compile(code, '<code>', 'exec'))
        execs.append(compile(test, '<test>', 'exec'))
    except Exception:
        return {'passed': False, 'reason': 'compile_error'}

    vars = {}
    for e in execs:
        try:
            exec(e, vars)
        except Exception:
            return {'passed': False, 'reason': 'runtime_error'}

    return {'passed': True, 'reason': ''}


def extract_function_calls(code):
    function_call = ''
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_call = ast.unparse(node)
            break
    
    return function_call

def assert2call(stmt):
    return  extract_function_calls(stmt)
