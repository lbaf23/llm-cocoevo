import subprocess
from utils import *
from typing import *
import os
import uuid
import re
import multiprocessing
from multiprocessing import Pool, Queue


def extract_entry_point_method(code: str, entry_point: str) -> str:
    lines = code.splitlines()
    method_pattern = re.compile(rf'^\s*def\s+{re.escape(entry_point)}\s*\(')
    method_lines = []
    inside_method = False
    for line in lines:
        if method_pattern.match(line):
            inside_method = True
            method_lines.append(line)
        elif inside_method:
            if line.strip().startswith("def "):
                break
            method_lines.append(line)
    if method_lines:
        return "\n".join(method_lines)
    return ''


def replace_lines(s1: str, s2: str, start_line: int, end_line: int) -> str:
    lines = s1.splitlines()
    new_lines = s2.splitlines()
    lines[start_line - 1 : end_line] = new_lines
    return "\n".join(lines)


def execute_i(id: int, run_dir: str, file_name: str, time_limit: int, q: Queue) -> None:
    report_name = f'report_{id}.json'
    result = subprocess.run(
        ['timeout', f'{time_limit}', 'pytest', '--json-report', f'--json-report-file={report_name}', file_name],
        capture_output=True,
        text=True,
        cwd=run_dir
    )
    if os.path.exists(os.path.join(run_dir, report_name)):
        report = read_json(os.path.join(run_dir, report_name))
        passed_count = report['summary'].get('passed', 0)
        total_count = report['summary']['total']
    else:
        passed_count = 0
        total_count = 0
    if result.returncode == 0:
        passed = True
    else:
        passed = False
    q.put({
        'id': id,
        'passed': passed,
        'total_i': total_count,
        'passed_i': passed_count,
    })


def evaluate_real_world(
        code: str,
        env_type: str,
        data_args: Dict[str, Any],
        test_cases: List[str],
        num_process: int,
        total_time_limit: float,
) -> Dict[str, Union[float, List[bool], Dict[int, str]]]:
    """
    Args:
        code (str): 
        test_cases (List): 
        total_time_limit (float): total time limit

    Returns:
        Dict:
            score:
            status
    """


    if env_type == 'real_world_method':
        program = data_args['program']
        start_line = data_args['start_line']
        end_line = data_args['end_line']
        test_prefix = data_args['test_prefix']
        entry_point = data_args['entry_point']

        target_method = extract_entry_point_method(code, entry_point)
        target_code = replace_lines(program, target_method, start_line, end_line)

        code_run = target_code + '\n\n' + test_prefix
    else:
        context_program = data_args['context_program']
        context_test_program = data_args['context_test_program']
        code_run = context_program + '\n\n\n' + code + '\n\n\n' + context_test_program + '\n\n\n'


    run_dir = f'tmp-{uuid.uuid1()}'
    create_dirs(run_dir)
    total = len(test_cases)

    # save code to disk
    file_name_list = []
    for i in range(total):
        file_name = f'test_{i}.py'
        all_code = code_run + '\n\n' + test_cases[i]
        write_file(os.path.join(run_dir, file_name), all_code)
        file_name_list.append(file_name)

    # execute and get result
    num_process = min(num_process, multiprocessing.cpu_count())
    num_process = min(num_process, total)
    m = multiprocessing.Manager()
    q = m.Queue(maxsize=1000000)
    pool = Pool(processes=num_process)
    pool_res = pool.starmap_async(
        execute_i,
        [(i, run_dir, file_name_list[i], total_time_limit, q) for i in range(total)],
    )
    pool_res.get()

    results = {}
    while not q.empty():
        res = q.get(False)
        results[res['id']] = res

    m.shutdown()
    pool.close()

    for i in range(total):
        if not results.__contains__(i):
            results[i] = {'id': i, 'passed': False, 'passed_i': 0, 'total_i': 1, 'reason': 'timeout_error'}

    # read result
    status = []

    passed = 0
    passed_count = 0
    total_count = 0

    for i in range(total):
        if results[i]['passed']:
            status.append(True)
            passed += 1
        else:
            status.append(False)

        passed_count += results[i]['passed_i']
        total_count += results[i]['total_i']

    delete_dirs(run_dir)

    # calculate score
    score = 0 if total == 0 else passed / total
    ret = {
        'score': score,
        'passed': passed,
        'total': total,
        'feedbacks': [],

        'passed_count': passed_count,
        'total_count': total_count,
        'status': status,
    }

    return ret
