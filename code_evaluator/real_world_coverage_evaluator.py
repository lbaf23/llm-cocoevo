import subprocess
from utils import *
from typing import *
import os
import uuid


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


def get_real_world_line_cov_feedback(
        code: str,
        test_cases: List[str],
        data_args: Dict[str, Any],
        total_time_limit: float = 10,
) -> Dict[str, Union[str, float]]:
    """
    Args:
        code (str): 
        test_cases (List): 
        total_time_limit (float): total time limit

    Returns:
        Dict[str, Union[str, float]]: Coverage and feedback
            {
                "coverage":
                "feedback":
            }
    """
    code = code.strip()

    context_program = data_args['context_program']
    context_test_program = data_args['context_test_program']
    code_run = context_program + '\n\n\n' + code + '\n\n\n' + context_test_program + '\n\n\n'


    run_dir = f'tmp-{uuid.uuid1()}'
    create_dirs(run_dir)
    total = len(test_cases)

    file_name = 'test_function.py'
    all_code = code_run + '\n\n' + '\n\n\n'.join(test_cases)
    write_file(os.path.join(run_dir, file_name), all_code)

    start_line = context_program.count('\n') + 4

    if total > 0:
        cmd = [
            'timeout',
            f'{total_time_limit}',
            'pytest',
            f'--cov={file_name[: -3]}',
            '--cov-report=json',
            file_name
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=run_dir
        )

        if os.path.exists(os.path.join(run_dir, 'coverage.json')):
            cov_report = read_json(os.path.join(run_dir, 'coverage.json'))
            missing_lines = cov_report['files'][file_name]['missing_lines']
            covered_lines = set()
            for i in range(0, code.count('\n') + 1):
                covered_lines.add(i)

            for line in missing_lines:
                if line >= start_line:
                    if covered_lines.__contains__(line - start_line):
                        covered_lines.remove(line - start_line)
        else:
            covered_lines = set()
    else:
        covered_lines = set()

    delete_dirs(run_dir)

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
