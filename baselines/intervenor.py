"""
INTERVENOR

https://github.com/NEUIR/INTERVENOR

"""

from typing import List, Dict, Tuple
from utils import extract_code, APIModels
import argparse
import os
from jsonl_utils import read_jsonl, append_jsonl
from file_utils import create_dirs
from log_utils import init_log, print_log
from tqdm import tqdm


def code_generation(model, prompt) -> Dict:
    system_prompt = 'You are expert programmer.'
    user_prompt = f'''\
Please write the implementation of the provided function in a code block.
```python
{prompt}
```
'''
    gen = model.generate_chat(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )

    output = gen['output']
    tokens_conut = gen['tokens_count']

    print_log(f'system_prompt', system_prompt, 0)
    print_log(f'user_prompt', user_prompt, 0)
    print_log(f'output', output, 0)

    code = extract_code(output)

    print_log(f'extract code', output, 0)

    return {
        'output': code,
        'tokens_count': tokens_conut
    }


def cor_generation(model, prompt, buggy_code, error_message) -> Dict:
    system_prompt = 'You are an experienced and insightful programming instructor, and you need to identify the bugs in the given code based on the error messages.'
    user_prompt = f'''\
- buggy code:
```python
{buggy_code}
```

When testing the above code, errors occurred: {error_message}, some test cases did not pass!
Please check the implementation of the function and provide a method for modification based on the error message. No need to provide the modified code.

Modification method:
'''
    
    gen = model.generate_chat(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        max_tokens=256
    )

    output = gen['output']
    tokens_conut = gen['tokens_count']

    print_log(f'cor system_prompt', system_prompt, 0)
    print_log(f'cor user_prompt', user_prompt, 0)
    print_log(f'cor output', output, 0)

    cor = output.strip()
    return {
        'output': cor,
        'tokens_count': tokens_conut
    }


def code_repairing(model, buggy_code, error_message, repair_method) -> Dict:
    system_prompt = 'You are a student assistant with excellent code repair capabilities. You can attempt to fix the bugs in the above code based on the provided error information and the method for modification. Please make sure to carefully check every potentially problematic area and make appropriate adjustments and corrections. Write your result in a code block.'
    user_prompt = f'''\
- buggy code:
```python
{buggy_code}
```

When testing the above code, errors occurred: {error_message} , some test cases did not pass! Please check the implementation of the function and fix the code based on the modification method.

- modification method:
{repair_method}

Correct the code:
'''    
    
    gen = model.generate_chat(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    output = gen['output']
    tokens_count = gen['tokens_count']

    print_log(f'repair system_prompt', system_prompt, 0)
    print_log(f'repair user_prompt', user_prompt, 0)
    print_log(f'repair output', output, 0)

    code = extract_code(output)

    print_log(f'extract code', code, 0)

    return {
        'output': code,
        'tokens_count': tokens_count
    }


def load_tests(file_path: str) -> List[str]:
    tests = []
    tests_set = set()
    content = read_jsonl(file_path)
    content = content[:10]
    for c in content:
        tests_i = c['tests'][:10]
        for t in tests_i:
            if not tests_set.__contains__(t['test']):
                tests_set.add(t['test'])
                tests.append(t['test'])
    return tests


import sys
from multiprocessing import Process, Queue
from io import StringIO
def evaluate(code: str, test_cases: List[str]) -> Tuple[float, str]:
    if len(test_cases) == 0:
        return 0.0, ''

    error_message = ''

    # Check for syntax errors first
    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        error_message = f"{type(e).__name__} - {str(e)}"

    def execute(code: str, test_cases: List[str], q: Queue):
        # Execute the code if no syntax errors
        error_message = ''

        std_out = sys.stdout
        sys.stdout = StringIO()

        for test in test_cases:
            try:
                exec_vars = {}
                exec(code + '\n\n' + test, exec_vars)
            except AssertionError as e:
                error_message = f"AssertionError - Assertion failed for {test}"
                break
            except Exception as e:
                error_message = f"{type(e).__name__} - {str(e)} for {test}"
                break

        sys.stdout = std_out
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

    parser.add_argument('--tests_dir', type=str)  # self-generted test cases
    args = parser.parse_args()

    model = APIModels(name='api', model_path=args.model_path, api_key=args.api_key, base_url=args.base_url)

    dataset = read_jsonl(args.dataset_path)
    iterator_rounds = 100


    for i in range(0, len(dataset)):
        if args.end > args.start and not (args.start <= i < args.end):
            continue

        create_dirs(args.save_dir)
        create_dirs(os.path.join(args.save_dir, 'logs'))

        result_file = os.path.join(args.save_dir, f'result_{i}.jsonl')
        init_log(os.path.join(args.save_dir, 'logs', f'log_{i}.log'))

        gen_test_file = os.path.join(args.tests_dir, f'result_{i}.jsonl')
        prompt = dataset[i]['prompt']

        r = 0
        code = ''
        cor = ''
        tests = load_tests(gen_test_file)
        assert len(tests) > 0
        
        if os.path.exists(result_file):
            content = read_jsonl(result_file)

            r = content[-1]['r'] + 1
            code = content[-1]['code']
            error_message = content[-1]['error_message']

            if content[-1]['score'] == 1.0 or len(tests) == 0:
                continue

        td = tqdm(initial=r, total=iterator_rounds)
        td.set_description(f'''[{i}]''')

        while r < iterator_rounds:
            if r == 0:
                # generate 
                gen = code_generation(model, prompt)
                code = gen['output']
                tokens_count = gen['tokens_count']
                score, error_message = evaluate(code, tests)
            else:
                assert error_message != ''
                gen = cor_generation(
                    model,
                    prompt=prompt,
                    buggy_code=code,
                    error_message=error_message
                )

                cor = gen['cor']
                tokens_count = gen['tokens_count']

                gen = code_repairing(
                    model,
                    buggy_code=code,
                    error_message=error_message,
                    repair_method=cor
                )
                code = gen['output']
                tokens_count2 = gen['tokens_count']
                tokens_count['prompt_tokens'] += tokens_count2['prompt_tokens']
                tokens_count['completion_tokens'] += tokens_count2['completion_tokens']

                score, error_message = evaluate(code, tests)

            result = {
                'r': r,
                'code': code,
                'error_message': error_message,
                'score': score,
                'cor': cor,
                'all_tokens_count': tokens_count
            }
            append_jsonl(result_file, result)

            r += 1
            td.update(1)
            if score == 1.0:
                break

