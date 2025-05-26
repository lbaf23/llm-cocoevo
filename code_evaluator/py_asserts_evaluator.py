from utils import filter_long_text
from utils import StdUtils
from multiprocessing import Process, Queue, Pool
import multiprocessing
from .utils import assert2call
from typing import *


def get_output_i(id: int, code: str, test_case: str, q: Queue) -> None:
    std_utils = StdUtils()
    std_utils.redirect_str_io()

    call = assert2call(test_case)
    code = code + '\n\n' + f'function_return_value = {call}'
    try:
        vars = {}
        exec(code, vars)

        std_utils.recover()
        if vars.__contains__('function_return_value'):
            return_value = filter_long_text(vars['function_return_value'])
            q.put({'id': id, 'output': return_value})
        else:
            q.put({'id': id, 'output': ''})
    except Exception as e:
        std_utils.recover()
        q.put({'id': id, 'output': str(e)})


def execute_i(id: int, code: str, test: str, q: Queue):
    std_utils = StdUtils()
    std_utils.redirect_str_io()

    code = code + '\n\n' + test

    try:
        code_exec = compile(code, '<code>', 'exec')
    except Exception:
        std_utils.recover()
        q.put({'id': id, 'passed': False, 'reason': 'compile_error'})
        return 

    try:
        vars = {}
        exec(code_exec, vars)
    except Exception:
        std_utils.recover()
        q.put({'id': id, 'passed': False, 'reason': 'runtime_error'})
        return

    std_utils.recover()
    q.put({'id': id, 'passed': True, 'reason': ''})


def evaluate_py_asserts_multi_process(
        code: str,
        test_cases: List[str],
        total_time_limit: float,
        num_process: int,
        feedback: bool = True
) -> Dict[str, Union[float, List[bool], Dict[int, str]]]:
    """
    Args:
        code (str): 
        test_cases (List): 
        total_time_limit (float): total time limit
        num_process (int): number of process
        feedback (bool): whether generate test case feedback

    Returns:
        Dict:
            feedbacks: [
                {
                    "id": 1,
                    "test": "assert ...",
                    "message": "",
                    "program_output": "",
                }
            ]
    """

    total = len(test_cases)

    num_process = min(num_process, multiprocessing.cpu_count())
    num_process = min(num_process, total)

    m = multiprocessing.Manager()
    q = m.Queue(maxsize=1000000)
    pool = Pool(processes=num_process)
    pool_res = pool.starmap_async(
        execute_i,
        [(i, code, test_cases[i], q) for i in range(total)],
    )

    results = {}
    try:
        pool_res.get(total_time_limit)
    except Exception:
        pool.terminate()
        pool.join()

    while not q.empty():
        res = q.get(False)
        results[res['id']] = res

    m.shutdown()
    pool.close()

    for i in range(total):
        if not results.__contains__(i):
            results[i] = {'passed': False, 'reason': 'timeout_error'}

    passed = 0
    status = []
    feedbacks = [{
        'id': i,
        'test': test_cases[i]
    } for i in range(total)]

    get_output_ids = []
    for i in range(total):
        result = results[i]

        if result['passed']:
            status.append(True)
            passed += 1
            if feedback:
                feedbacks[i]['passed'] = False
        else:
            status.append(False)
            if feedback:
                feedbacks[i]['passed'] = False
                if result['reason'] == 'timeout_error':
                    message = f'{filter_long_text(test_cases[i])}  # program output: timeout error'
                    program_output = 'timeout error'
                    feedbacks[i]['message'] = message
                    feedbacks[i]['program_output'] = program_output
                else:
                    get_output_ids.append(i)

    # calculate program output
    if feedback and len(get_output_ids) > 0:
        num_process = min(num_process, len(get_output_ids))
        m = multiprocessing.Manager()
        q = m.Queue(maxsize=1000)
        pool = Pool(processes=num_process)
        pool_res = pool.starmap_async(
            get_output_i,
            [(id, code, test_cases[id], q) for id in get_output_ids],
        )

        outputs = {}
        try:
            pool_res.get(total_time_limit)
        except Exception:
            pool.terminate()
            pool.join()
        while not q.empty():
            res = q.get(False)
            outputs[res['id']] = res

        m.shutdown()
        pool.close()

        for id in get_output_ids:
            if outputs.__contains__(id):
                message = f'''{filter_long_text(test_cases[id])}  # program output: {outputs[id]['output']}'''
                program_output = outputs[id]['output']
            else:
                message = f'{filter_long_text(test_cases[id])}'
                program_output = ''

            feedbacks[id]['message'] = message
            feedbacks[id]['program_output'] = program_output

    # calculate score
    score = 0 if total == 0 else passed / total
    ret = {
        'score': score,
        'passed': passed,
        'total': total,
        'passed_count': passed,
        'total_count': total,
        'status': status
    }

    if feedback:
        ret['feedbacks'] = feedbacks
    else:
        ret['feedbacks'] = []
    return ret
