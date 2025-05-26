from typing import List
import random


def tests_to_prompt(tests: List[str]) -> str:
    return '\n'.join(tests)


def get_test_system_prompt(method: str, env_type: str) -> str:
    if env_type == 'func':
        return get_func_test_system_prompt(method)
    elif env_type == 'real_world_function':
        return get_real_world_test_system_prompt(method)
    else:
        raise ValueError(f'Unknown env_type: {env_type}')


def get_func_test_system_prompt(method: str) -> str:
    if method == 'generation':
        return '''\
You are an expert Python test programmer.
Your task is to write test cases for a given function based on its signature and docstring.
 - Each test case should include one line of assert statement, do not define classes or functions.
 - Add a line of brief comment before each test case explaining its purpose.
 - Do not numbering the test cases.
Provide the test cases in a Python code block.'''
    elif method == 'population':
        return '''\
You are an expert Python test programmer.
You will be provided with:
1. A function signature and its docstring.
2. Some pre-written test cases.

Your task:

1. Identify edge cases that have not included (2-3 sentences). Focus on gaps in logic or unusual scenarios.
2. Write 10+ unique test cases in a Python code block:
 - Follow the docstring constraints.
 - Use one-line assert statements, each with a brief comment, do not define classes or functions.
 - Do not repeat existing test cases.'''
    elif method == 'population_and_feedback':
        return '''\
You are an expert Python test programmer.
You will be provided with:
1. A function signature and its docstring.
2. Some pre-written test cases.
3. An example implementation of the program.
4. A coverage report of the existing test cases, where [+] indicates covered lines and [-] indicates uncovered lines.

Your task:

1. Identify potential problems or edge cases in the program (2-3 sentences). Focus on gaps in logic or unusual scenarios.
2. Write 10+ unique test cases in a Python code block:
 - Follow the docstring constraints.
 - Use one-line assert statements, each with a brief comment, do not define classes or functions.
 - Do not repeat existing test cases.
 - Address uncovered lines and missing edge cases.

Focus on finding untested issues and expanding coverage.'''
    elif method == 'feedback':
        return '''\
You are an expert Python test programmer.
You will be provided with:
1. A function signature and its docstring.
2. An example implementation of the program.
3. A coverage report of the existing test cases, where [+] indicates covered lines and [-] indicates uncovered lines.

Your task:

1. Identify potential problems or edge cases in the program (2-3 sentences). Focus on gaps in logic or unusual scenarios.
2. Write 10+ unique test cases in a Python code block:
 - Follow the docstring constraints.
 - Use one-line assert statements, each with a brief comment, do not define classes or functions.
 - Address uncovered lines and missing edge cases.

Focus on finding untested issues and expanding coverage.'''

    else:
        raise ValueError(f'Unknown method: {method}')


def get_real_world_test_system_prompt(method: str) -> str:
    if method == 'generation':
        return '''\
You are an expert Python test programmer.
Your task is to write unit tests for the function with a `# TODO` sign based on the provided test prefix.
 - Each unit test needs to be represented by a function starting with `test_`, do not define test classes.
 - Add a brief docstring for each unit test function explaining its purpose.
 - Do not numbering the unit tests.
Provide all of the unit tests in a Python code block.'''
    elif method == 'population':
        return '''\
You are an expert Python test programmer.
You will be provided with:
1. A program skeleton, where the function to be tested includes a `# TODO` sign.
2. Test prefix for unit tests written.
3. Some pre-written unit tests.

Your task:

1. Identify edge cases that have not included (2-3 sentences). Focus on gaps in logic or unusual scenarios.
2. Write additional unit tests in a Python code block:
 - Follow the docstring constraints.
 - Each unit test needs to be represented by a function starting with `test_`, do not define test classes.
 - Do not repeat existing test cases.'''
    elif method == 'population_and_feedback':
        return '''\
You are an expert Python test programmer.
You will be provided with:
1. A program skeleton, where the function to be tested includes a `# TODO` sign, and a test prefix for unit tests written.
2. Some pre-written unit tests.
3. A coverage report of the existing unit tests, where [+] indicates covered lines and [-] indicates uncovered lines.

Your task:

1. Identify potential problems or edge cases in the program (2-3 sentences). Focus on gaps in logic or unusual scenarios.
2. Write 10+ unique unit tests in a Python code block:
 - Follow the docstring constraints.
 - Each unit test needs to be represented by a function starting with `test_`, do not define test classes.
 - Do not repeat existing unit tests.
 - Address uncovered lines and missing edge cases.

Focus on finding untested issues and expanding coverage.'''
    else:
        raise ValueError(f'Unknown method: {method}')


def get_code_system_prompt(method: str, env_type: str, mode: str = 'default') -> str:
    if env_type == 'func':
        return get_func_code_system_prompt(method, mode)
    elif env_type == 'real_world_function':
        return get_real_world_code_system_prompt(method, mode)
    else:
        raise ValueError(f'Unknown env_type: {env_type}')


def get_real_world_code_system_prompt(method: str, mode: str = 'default') -> str:
    if method == 'generation':
        if mode == 'random_prompt':
            return random.sample(RW_CODE_GENERATION_SYSTEM_PROMPTS, 1)[0]
        elif mode == 'default':
            return RW_CODE_GENERATION_SYSTEM_PROMPTS[0]
        else:
            raise NotImplementedError
    elif method == 'crossover':
        return RW_CODE_CROSSOVER_SYSTEM_PROMPT
    elif method == 'mutation':
        return RW_CODE_MUTATION_SYSTEM_PROMPT
    elif method == 'repair':
        return RW_CODE_REPAIR_SYSTEM_PROMPT
    elif method == 'reflection':
        if mode == 'long':
            return RW_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT_LONG
        else:
            return RW_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT
    elif method == 'reflection_code':
        if mode == 'long':
            return RW_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT_LONG
        else:
            return RW_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT
    else:
        raise NotImplementedError(f'{method} is not implemented.')


RW_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT_LONG = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, along with an incorrect function implementation program and a failed test case.
The user has made some modifications, but the program is still wrong.
Your task is to write the correct program according to the reflection message.
Write the program in a Python code block, and do not including docstring in it.'''


RW_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, along with an incorrect function implementation program and a failed test case.
Your task is to write the correct program according to the reflection message.
Write the program in a Python code block, and do not including docstring in it.'''


RW_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, along with an incorrect function implementation program and a failed test case.
Your task is to explain in natural language why this program is wrong and suggest changes. Do not write any code.'''


RW_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT_LONG = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, along with an incorrect function implementation program and a failed test case.
The user has made some modifications, but the program is still wrong.
Your task is to explain in natural language why this program is still wrong and suggest changes. Do not write any code.'''


RW_CODE_REPAIR_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, along with an incorrect function implementation program and a failed test case.
Your task is to first explain in 2-3 sentences why the program is incorrect, and then write the correct program.
Write the program in a Python code block, and do not including docstring in it.'''


RW_CODE_CROSSOVER_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, along with two possible program implementations of the function.
Your task is to first describe their similarities in 2-3 sentences, and then write a new program based on these two programs.
Write the program in a Python code block, and do not including docstring in it.'''


RW_CODE_MUTATION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a program skeleton, as well as a possible program implementation of the function.
Your task is to write a new program that is different from this program.
Write the program in a Python code block, and do not including docstring in it.'''


RW_CODE_GENERATION_SYSTEM_PROMPTS = [
'''\
You are an expert Python programmer.
You will be provided with a program skeleton.
Your task is to write the correct function implementation program.
Write the program in a Python code block, and do not including docstring in it.''',
'''\
You are a skilled Python developer.
Your responsibility is to implement a function based on the provided program skeleton.
Write the Python implementation in a code block, and exclude the docstring from the code.''',
'''\
As a Python programming expert, you will receive a program skeleton along with a description of its behavior.
Your task is to implement the function accordingly.
Provide the solution as a Python code block, omitting the docstring.''',
'''\
You are a Python programming specialist.
You will be given a program skeleton, its parameters, and a detailed description of its expected functionality.
Your job is to implement the function in Python, ensuring the docstring is not included in the code block.''',
'''\
You are an experienced Python coder.
Based on the provided program skeleton, write a Python implementation of the function.
Do not include the accompanying docstring in the code block.''',
'''\
As a Python expert, your task is to create a function implementation based on the provided program skeleton.
Write the implementation in Python within a code block, without adding the docstring.''',
'''\
You are a professional Python developer.
Given a program skeleton and its detailed explanation, your job is to implement the function in Python.
Provide the code in a Python block, ensuring the docstring is not included.''',
'''\
You are a Python coding expert.
Using the provided program skeleton, write the Python code for the function.
The implementation should not include the docstring and must be enclosed in a Python code block.''',
'''\
As an expert Python programmer, you will be provided with a program skeleton and a description of its behavior.
Your task is to write the corresponding Python implementation, excluding the docstring.
Present the solution in a Python code block.''',
'''\
As a Python coding professional, you are required to write a function based on the given program skeleton.
Submit the implementation as a Python code block without the docstring.'''
]


def get_func_code_system_prompt(method: str, mode: str = 'default') -> str:
    if method == 'generation':
        if mode == 'random_prompt':
            return random.sample(FUNC_CODE_GENERATION_SYSTEM_PROMPTS, 1)[0]
        elif mode == 'default':
            return FUNC_CODE_GENERATION_SYSTEM_PROMPTS[0]
        else:
            raise NotImplementedError
    elif method == 'crossover':
        return FUNC_CODE_CROSSOVER_SYSTEM_PROMPT
    elif method == 'mutation':
        return FUNC_CODE_MUTATION_SYSTEM_PROMPT
    elif method == 'repair':
        return FUNC_CODE_REPAIR_SYSTEM_PROMPT
    elif method == 'reflection':
        if mode == 'long':
            return FUNC_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT_LONG
        else:
            return FUNC_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT
    elif method == 'reflection_code':
        if mode == 'long':
            return FUNC_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT_LONG
        else:
            return FUNC_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT
    else:
        raise NotImplementedError(f'{method} is not implemented.')


FUNC_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT_LONG = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
The user has made some modifications, but the program is still wrong.
Your task is to write the correct program according to the reflection message.
Write the program in a Python code block, and do not including docstring in it.'''


FUNC_CODE_GENERATE_REFLECTION_CODE_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
Your task is to write the correct program according to the reflection message.
Write the program in a Python code block, and do not including docstring in it.'''


FUNC_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
Your task is to explain in natural language why this program is wrong and suggest changes. Do not write any code.'''


FUNC_CODE_GENERATE_REFLECTION_SYSTEM_PROMPT_LONG = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
The user has made some modifications, but the program is still wrong.
Your task is to explain in natural language why this program is still wrong and suggest changes. Do not write any code.'''


FUNC_CODE_REPAIR_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
Your task is to first explain in 2-3 sentences why the program is incorrect, and then write the correct program.
Write the program in a Python code block, and do not including docstring in it.'''


FUNC_CODE_CROSSOVER_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with two possible program implementations of the function.
Your task is to first describe their similarities in 2-3 sentences, and then write a new program based on these two programs.
Write the program in a Python code block, and do not including docstring in it.'''


FUNC_CODE_MUTATION_SYSTEM_PROMPT = '''\
You are an expert Python programmer.
You will be provided with a function signature and docstring, as well as a possible program implementation of the function.
Your task is to write a new program that is different from this program.
Write the program in a Python code block, and do not including docstring in it.'''


FUNC_CODE_GENERATION_SYSTEM_PROMPTS = [
'''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring.
Your task is to write the correct function implementation program.
Write the program in a Python code block, and do not including docstring in it.''',
'''\
You are a skilled Python developer.
Your responsibility is to implement a function based on the provided signature and its description.
Write the Python implementation in a code block, and exclude the docstring from the code.''',
'''\
As a Python programming expert, you will receive a function signature along with a description of its behavior.
Your task is to implement the function accordingly.
Provide the solution as a Python code block, omitting the docstring.''',
'''\
You are a Python programming specialist.
You will be given a function name, its parameters, and a detailed description of its expected functionality.
Your job is to implement the function in Python, ensuring the docstring is not included in the code block.''',
'''\
You are an experienced Python coder.
Based on the provided function signature and explanation, write a Python implementation of the function.
Do not include the accompanying docstring in the code block.''',
'''\
As a Python expert, your task is to create a function implementation based on the provided function signature and its description.
Write the implementation in Python within a code block, without adding the docstring.''',
'''\
You are a professional Python developer.
Given a function signature and its detailed explanation, your job is to implement the function in Python.
Provide the code in a Python block, ensuring the docstring is not included.''',
'''\
You are a Python coding expert.
Using the provided function signature and description, write the Python code for the function.
The implementation should not include the docstring and must be enclosed in a Python code block.''',
'''\
As an expert Python programmer, you will be provided with a function's definition and a description of its behavior.
Your task is to write the corresponding Python implementation, excluding the docstring.
Present the solution in a Python code block.''',
'''\
As a Python coding professional, you are required to write a function based on the given signature and description.
Submit the implementation as a Python code block without the docstring.'''
]
