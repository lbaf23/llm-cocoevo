from typing import List, Dict, Set, Union, Any
import multiprocessing
from multiprocessing import Pool, Queue
from utils import try_format_code, StdUtils
import sys


def add_cov_message(
        code: str,
        lines_not_covered: Union[List, Set],
        covered_c: str = '+',
        not_covered_c: str = '-'
) -> str:
    res = ''
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if lines_not_covered.__contains__(i):
            res += f'[{not_covered_c}] ' + line + '\n'
        else:
            res += f'[{covered_c}] ' + line + '\n'
    return res.rstrip()


def cover(id: int, code: str, test: str, q: Queue) -> None:
    std_utils = StdUtils()
    std_utils.redirect_str_io()

    execs = []
    try:
        execs.append(compile(code, '<string-code>', 'exec'))
        execs.append(compile(test, '<string-test>', 'exec'))
    except Exception:
        std_utils.recover()
        q.put({'id': id, 'passed': False, 'covered_lines': [], 'reason': 'compile_error'})
        return

    vars = {}
    trace_lines = set()

    def trace_function(frame, event, arg=None):
        file_name = frame.f_code.co_filename
        line_no = frame.f_lineno
        if file_name != '<string-code>':
            return
        if event == 'line':
            trace_lines.add(int(line_no) - 1)
        return trace_function

    sys.settrace(trace_function)
    passed = False
    try:
        for e in execs:
            exec(e, vars)
        passed = True
        reason = ''
    except Exception:
        reason = 'runtime_error'
    sys.settrace(None)

    std_utils.recover()
    q.put({'id': id, 'passed': passed, 'covered_lines': trace_lines, 'reason': reason})


def get_py_asserts_line_cov_feedback(
        code: str,
        test_cases: List[str],
        env_type: str,
        data_args: Dict[str, Any],
        num_process: int = 5,
        total_time_limit: float = 2.0,
) -> Dict[str, Union[str, float]]:
    """
    Args:
        code: Python code
        test_cases: List of test cases
        env_type: Type of evaluator, func, repo_exec
        data_args:
        num_process: Number of processes
        total_time_limit: Total time limit

    Returns:
        Dict[str, Union[str, float]]: Coverage and feedback
            {
                "coverage":
                "feedback":
            }

    """

    code = try_format_code(code, mode='hard')

    if env_type == 'repo_exec':
        assert data_args.__contains__('check_prefix')
        check_prefix = data_args['check_prefix']

        exec_code = check_prefix + '\n\n\n' + code
        start_line = check_prefix.count('\n') + 3
    elif env_type == 'func':
        start_line = 0
        exec_code = code
    else:
        raise NotImplementedError(f'No such env_type: {env_type}')

    total = len(test_cases)

    if total > 0:
        num_process = min(num_process, multiprocessing.cpu_count())
        num_process = min(num_process, total)

        m = multiprocessing.Manager()
        q = m.Queue(maxsize=1000000)
        pool = Pool(processes=num_process)

        pool_res = pool.starmap_async(
            cover,
            [(i, exec_code, test_cases[i], q) for i in range(total)],
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
        covered_lines = set()
        for res in results.values():
            for line in res['covered_lines']:
                if line >= start_line:
                    covered_lines.add(line - start_line)

    else:
        covered_lines = set()

    # calculate line coverage
    lines = code.split('\n')
    covered = 0
    all_lines = 0
    line_no = set()
    for i, l in enumerate(lines):
        l = l.strip()
        if l != 'else:' and \
                l != '' and \
                not (l.startswith('\'') and l.endswith('\'')) and \
                not l.strip().startswith('#'):
            if covered_lines.__contains__(i):
                covered += 1
                all_lines += 1
            else:
                line_no.add(i)
                all_lines += 1
    coverage = covered / all_lines if all_lines > 0 else 0
    lines_not_covered = add_cov_message(code, line_no)

    return {
        'coverage': coverage,
        'feedback': lines_not_covered
    }
