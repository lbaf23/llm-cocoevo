"""
CodeCOT

"""

from typing import List, Tuple
from utils import extract_code, APIModels
import argparse
import os
from jsonl_utils import read_jsonl, append_jsonl
from file_utils import create_dirs
from log_utils import init_log, print_log
from tqdm import tqdm


code_generation_prompt = '''\
### Instruction:
Please write the program and test cases based on the task description.

For example:

### Task Description:
```python
def factorial(n):
    """
    Return the factorial of n.
    >>> factorial(2)
    2
    >>> factorial(0)
    1
    """
```

### Chain of Thought Reasoning:
1. **Understanding Factorial**: Recognize that the factorial of a number `n` is the product of all positive integers
from 1 to `n`.
2. **Choosing the Approach**: Decide between iterative and recursive approaches. Opt for the iterative approach to
avoid recursion limit issues for larger numbers.
3. **Implementing the Iterative Approach**: Start with initializing a result variable to 1. Then, multiply it sequentially
with every integer from 1 to `n`.
4. **Handling Edge Case**: Account for the edge case where `n` is 0. By definition, 0! (0 factorial) equals 1.
5. **Testing the Function**: After implementation, test the function with different values of `n` to ensure its
correctness.


### Code Implementation:
```python
def factorial(n):
    """
    Return the factorial of n.
    """
    # Handle the edge case for 0 factorial
    if n == 0:
    return 1
    # Initialize the result variable
    result = 1
    # Iteratively compute the factorial
    for i in range(1, n + 1):
    result *= i
    return result
```

### Self-examination with Test Cases:
```python
# Testing the function with various cases
assert factorial(0)==1 # factorial(0) should return 1
assert factorial(0)==1 # factorial(1) should return 1
assert factorial(2)==2 # factorial(2) should return 2
assert factorial(5)==120 # factorial(5) should return 120
assert factorial(10)==3628800 # factorial(10) should return 3628800
```

### Task Description:
```python
{prompt}
```
'''


self_examination_with_feedback_prompt = '''\
### Instruction:
Below is a code snippet and its test cases. Please fix the bugs reported by the local environment, and write the fixed code in a Python code block directly, without any other content.

### Code Snippet:
```python
{error_code}
```

### Test Cases:
```python
{test_cases}
```

### Error Messages:
```python
{error_message}
```

### Fixed Code:
'''


def split_tests(output: str) -> List[List]:
    tests = []
    lines = output.split('\n')
    for l in lines:
        l = l.strip()
        if l.startswith('assert '):
            test = l.strip()
            description = ''
            if test.__contains__('#'):
                description = test[test.rindex('#') + 1 : ].strip()
                test = test[ : test.rindex('#')].strip()
            tests.append([test.strip(), description.strip()])

    return tests


def extract_cot(output: str) -> Tuple[str, str, List[List[str]]]:
    cot_start = '### Chain of Thought Reasoning:'
    code_start = '### Code Implementation:'
    test_start = '### Self-examination with Test Cases:'

    cot, code, tests = '', '', ''

    if output.__contains__(cot_start):
        content = output.split(cot_start)
        cot = content[0]
        output = content[1]
    
    if output.__contains__(code_start):
        cot, output = output.split(code_start)
        code = extract_code(output)
    else:
        code = extract_code(output)

    if output.__contains__(test_start):
        _, tests = output.split(test_start)
    else:
        tests = output

    tests = split_tests(tests)

    return cot.strip(), code, tests


from multiprocessing import Process, Queue

def execution(code: str, test_cases: List[List[str]]) -> Tuple[float, List[str]]:
    if len(test_cases) == 0:
        return 0.0, []

    errors = []
    # Check for syntax errors first
    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")


    def execute(compiled_code: str, test_cases: List[List[str]], q: Queue):
        # Execute the code if no syntax errors
        exec_vars = {}
        errors = []
        exec(compiled_code, exec_vars)
        for test, description in test_cases:
            try:
                exec(test, exec_vars)
            except AssertionError:
                error_message = f"AssertionError in test case: {test} due to {description}"
                # print(error_message)
                errors.append(error_message)
                break
            except Exception as e:
                error_message = f"Exception in test case: {test} due to {description}"
                errors.append(error_message)
                break
        q.put(errors)

    if not errors:
        q = Queue()
        p = Process(target=execute, args=(compiled_code, test_cases, q,))
        p.start()
        p.join(2.0)

        try:
            errors = q.get(block=False)
        except Exception:
            p.terminate()
            p.join()
            errors = ['Timeout error']

    if errors:
        return 0.0, errors
    else:
        return 1.0, []  #  "All tests passed successfully!"


def tests_to_str(tests: List[List[str]]) -> str:
    tests = [' # '.join(t) for t in tests]
    tests = '\n'.join(tests)
    return tests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()

    model = APIModels(name='api', model_path=args.model_path, api_key=args.api_key, base_url=args.base_url)

    dataset = read_jsonl(args.dataset_path)
    iterator_rounds = 100

    for i in range(len(dataset)):
        if args.end > args.start and not (args.start <= i < args.end):
            continue

        create_dirs(args.save_dir)
        create_dirs(os.path.join(args.save_dir, 'logs'))

        result_file = os.path.join(args.save_dir, f'result_{i}.jsonl')
        init_log(os.path.join(args.save_dir, 'logs', f'log_{i}.log'))

        prompt = dataset[i]['prompt']

        r = 0
        code = ''
        tests = []
        if os.path.exists(result_file):
            content = read_jsonl(result_file)

            r = content[-1]['r'] + 1
            code = content[-1]['code']
            tests = content[-1]['tests']
            errors = content[-1]['errors']

            if content[-1]['score'] == 1.0 or len(tests) == 0:
                continue

        td = tqdm(initial=r, total=iterator_rounds)
        td.set_description(f'''[{i}]''')

        while r < iterator_rounds:
            if r == 0:
                # generate 
                user_message = code_generation_prompt.format(
                    prompt=prompt
                )
                output = model.generate_chat(
                    [
                        {'role': 'user', 'content': user_message}
                    ],
                    stop_strs=['### Task Description:']
                )
                print_log(f'{r}: user_prompt', user_message, 0)
                print_log(f'{r}: output', output, 0)

                cot, code, tests = extract_cot(output)

                print_log(f'{r}: extract code', code, 0)
                print_log(f'{r}: extract tests', tests_to_str(tests), 0)

                score, errors = execution(code, tests)

            else:
                # self exam
                user_message = self_examination_with_feedback_prompt.format(
                    error_code=code,
                    test_cases=tests_to_str(tests),
                    error_message=errors[0]
                )
                output = model.generate_chat(
                    [
                        {'role': 'user', 'content': user_message}
                    ],
                    stop_strs=['### Code Snippet:', '### Test Cases:', '### Error Messages:']
                )
                print_log(f'{r}: user_prompt', user_message, 0)
                print_log(f'{r}: output', output, 0)

                code = extract_code(output)
                
                print_log(f'{r}: extract code', code, 0)

                score, errors = execution(code, tests)
            
            result = {
                'r': r,
                'code': code,
                'score': score,
                'errors': errors,
                'tests': tests
            }
            append_jsonl(result_file, result)

            r += 1
            td.update(1)
            if score == 1.0 or len(tests) == 0:
                break
