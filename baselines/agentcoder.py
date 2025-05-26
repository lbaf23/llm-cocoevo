"""
AgentCoder

https://github.com/huangd1999/AgentCoder

"""

from typing import List, Tuple
from utils import extract_code, APIModels, extract_asserts
import argparse
import os
from jsonl_utils import read_jsonl, append_jsonl
from file_utils import create_dirs
from log_utils import init_log, print_log
from tqdm import tqdm


programmer_prompt = '''\
As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
efficient, readable, and well-commented.

For example:

**Input Code Snippet**:
```python
{prompt}
    # Add your code here to complete the function
```

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient way.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code, write it in a Python code block.
'''


test_designer_prompt = '''\
**1. Basic Test Cases**:
- **Objective**: To verify the fundamental functionality of the `has_close_elements` function under normal
conditions.

**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

**3. Large Scale Test Cases**:
- **Objective**: To assess the function’s performance and scalability with large data samples.

**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.


For example:

**Input Code Snippet:**
```python
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**1. Basic Test Cases:**
```python
# Test 1: Standard list with some close elements
assert has_close_elements([1.0, 2.5, 3.5, 5.0], 1.0) == True
# Test 2: Standard list with no close elements
assert has_close_elements([1.0, 3.0, 5.0, 7.0], 1.5) == False
```

**2. Edge Test Cases:**
```python
# Test 1: Empty list
assert has_close_elements([], 1.0) == False

# Test 2: List with all elements the same
assert has_close_elements([3.0, 3.0, 3.0], 0.0) == True

# Test 3: Very small threshold
assert has_close_elements([1.0, 1.01, 2.0], 0.005) == False

# Test 4: List with only two elements
assert has_close_elements([1.0, 2.0], 1.5) == True
```

**3. Large Scale Test Cases:**
For large-scale testing, I'll focus on the function's performance with a significantly large list. Due to the constraints
of this platform, I'll conceptualize the test case:
```python
# Large Scale Test 1: List with 100,000 elements in a predictable pattern
large_list = [i * 0.1 for i in range(100000)] # Creates a list [0, 0.1, 0.2, ..., 9999.9]

# Test with a threshold where we know the outcome
# Since the list is in increments of 0.1, a threshold of 0.05 should return False
assert has_close_elements(large_list, 0.05) == False

# Test with a larger threshold where we expect a True result
# With a threshold of 0.15, adjacent elements (0.1 apart) will be within the threshold
assert has_close_elements(large_list, 0.15) == True
```

**Input Code Snippet**:
```python
{prompt}
```
'''


programmer_feedback_prompt = '''\
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


from multiprocessing import Process, Queue
import traceback

def evaluate(code: str, test_cases: List[str]) -> Tuple[float, str]:
    if len(test_cases) == 0:
        return 0.0, ''

    error_message = ''

    # Check for syntax errors first
    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        msg = traceback.format_exc()
        error_message = f"Syntax error: {e}\n{msg}"

    def execute(code: str, test_cases: List[str], q: Queue):
        # Execute the code if no syntax errors
        error_message = ''

        for test in test_cases:
            try:
                exec_vars = {}
                exec(code + '\n\n' + test, exec_vars)
            except AssertionError as e:
                msg = traceback.format_exc()
                error_message = f"AssertionError in test case: {test}\n{msg}"
                break
            except Exception as e:
                msg = traceback.format_exc()
                error_message = f"Exception in test case: {test}\n{msg}"
                break

        q.put(error_message)

    if error_message == '':
        q = Queue()
        p = Process(target=execute, args=(code, test_cases, q,))
        p.start()
        p.join(2.0)

        try:
            error_message = q.get(block=False)
        except Exception:
            p.terminate()
            p.join()
            error_message = 'Timeout error'

    if error_message != '':
        return 0.0, error_message
    else:
        return 1.0, ''  #  "All tests passed successfully!"


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
        error_message = ''
        if os.path.exists(result_file):
            content = read_jsonl(result_file)

            r = content[-1]['r'] + 1
            code = content[-1]['code']
            tests = content[-1]['tests']
            error_message = content[-1]['error_message']

            if content[-1]['score'] == 1.0 or len(tests) == 0:
                continue


        td = tqdm(initial=r, total=iterator_rounds)
        td.set_description(f'''[{i}]''')

        while r < iterator_rounds:
            if r == 0:
                # generate code

                user_message = programmer_prompt.format(
                    prompt=prompt
                )
                gen = model.generate_chat(
                    [
                        {'role': 'system', 'content': 'You are a software programmer.'},
                        {'role': 'user', 'content': user_message}
                    ],
                    stop_strs=[]
                )

                output = gen['output']
                tokens_count = gen['tokens_count']

                print_log(f'{r}: user_prompt: code', user_message, 0)
                print_log(f'{r}: output: code', output, 0)

                code = extract_code(output, '```python')

                print_log(f'{r}: extract code', code, 0)

                # generate tests
                user_message = test_designer_prompt.format(
                    prompt=prompt
                )
                system_prompt = '''As a tester, your task is to create comprehensive test cases for the incomplete function. These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability'''
                gen = model.generate_chat(
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message}
                    ],
                    stop_strs=[]
                )

                output = gen['output']
                tokens_count2 = gen['tokens_count']
                tokens_count['prompt_tokens'] += tokens_count2['prompt_tokens']
                tokens_count['completion_tokens'] += tokens_count2['completion_tokens']

                print_log(f'{r}: system_prompt: test', system_prompt, 0)
                print_log(f'{r}: user_prompt: test', user_message, 0)
                print_log(f'{r}: output: test', output, 0)

                tests = extract_asserts(output)

                print_log(f'{r}: extract tests', '\n'.join(tests), 0)

                score, error_message = evaluate(code, tests)
            else:
                assert error_message != ''
                system_prompt = 'You are a software programmer.'
                user_message = programmer_feedback_prompt.format(
                    error_code=code,
                    test_cases='\n'.join(tests),
                    error_message=error_message,
                )
                gen = model.generate_chat(
                    [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message}
                    ],
                    stop_strs=['### Code Snippet:', '### Test Cases:', '### Error Messages:']
                )
                output = gen['output']
                tokens_count = gen['tokens_count']

                print_log(f'{r}: system_prompt: repair', system_prompt, 0)
                print_log(f'{r}: user_prompt: repair', user_message, 0)
                print_log(f'{r}: output: repair', output, 0)

                code = extract_code(output, '```python')
                print_log(f'{r}: extract code', code, 0)

                score, error_message = evaluate(code, tests)
            
            result = {
                'r': r,
                'code': code,
                'score': score,
                'error_message': error_message,
                'tests': tests,
                'all_tokens_count': tokens_count
            }
            append_jsonl(result_file, result)

            r += 1
            td.update(1)
            if score == 1.0 or len(tests) == 0:
                break
