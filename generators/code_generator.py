from code_models import ModelBase
from utils import extract_code, print_log, add_block
from typing import Dict, Any
import random


system_prompt_list = [
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


class CodeGenerator:
    function_signature_and_docstring = '### Function Signature and Docstring'
    program = '### Program'

    new_program = '### New Program'

    program1 = '### Program 1'
    program2 = '### Program 2'

    description_and_new_program = '### Description and New Program'

    failed_test_case = '### Failed Test Case'
    incorrect_program = '### Incorrect Program'

    previous_code_and_modification = '### Previous Code and Modification'
    failed_test_case_of_previous_modification = '### Failed Test Case of Previous Modification'

    reflection_message = '### Reflection Message'

    explanation_and_correct_program = '### Explanation and Correct Program'
    previous_explanation_and_modification = '### Previous Explanation and Program'

    def __init__(
            self,
            model: ModelBase,
    ) -> None:
        self.model = model

    def generate(
            self,
            prompt: str,
            init_method: str = 'default',
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict:
        if init_method == 'random_prompt':
            system_prompt = random.sample(system_prompt_list, 1)[0]
        elif init_method == 'default':
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring.
Your task is to write the correct function implementation program.
Write the program in a Python code block, and do not including docstring in it.'''
        else:
            raise NotImplementedError

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.program}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']

        print_log('generate code [system]', system_prompt, 1)
        print_log('generate code [user]', user_prompt, 1)
        print_log('generate code [assistant]', output, 1)

        code = extract_code(output)
        print_log('generate code [code]', code, 1)

        return {
            'code': code,
            'output': output,
            'tokens_count': gen['tokens_count']
        }

    def generate_crossover(
            self,
            prompt: str,
            code1: str,
            code2: str,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:

        system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with two possible program implementations of the function.
Your task is to first describe their similarities in 2-3 sentences, and then write a new program based on these two programs.
Write the program in a Python code block, and do not including docstring in it.'''

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.program1}
{add_block(code1)}

{self.program2}
{add_block(code2)}

{self.description_and_new_program}
'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']

        print_log('generate code crossover [system]', system_prompt, 1)
        print_log('generate code crossover [user]', user_prompt, 1)
        print_log('generate code crossover [assistant]', output, 1)

        code = extract_code(output)
        print_log('generate code crossover [code]', code, 1)

        return {
            'code': code,
            'output': output,
            'tokens_count': gen['tokens_count']
        }

    def generate_mutation(
            self,
            prompt: str,
            code: str,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and docstring, as well as a possible program implementation of the function.
Your task is to write a new program that is different from this program.
Write the program in a Python code block, and do not including docstring in it.'''

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.program}
{add_block(code)}

{self.new_program}
'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']

        print_log('generate code mutation [system]', system_prompt, 1)
        print_log('generate code mutation [user]', user_prompt, 1)
        print_log('generate code mutation [assistant]', output, 1)

        code = extract_code(output)
        print_log('generate code mutation [code]', code, 1)

        return {
            'code': code,
            'output': output,
            'tokens_count': gen['tokens_count']
        }

    def generate_repair(
            self,
            prompt: str,
            code: str,
            test_feedback: str = '',
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        if test_feedback.strip() == '':
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with a possible function implementation program.
Your task is to first explain the possible errors in this program in 2-3 sentences, and then write the correct program.
Write the program in a Python code block, and do not including docstring in it.'''

            user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.program}
{add_block(code)}

{self.explanation_and_correct_program}
'''
        else:
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
Your task is to first explain in 2-3 sentences why the program is incorrect, and then write the correct program.
Write the program in a Python code block, and do not including docstring in it.'''

            user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.incorrect_program}
{add_block(code)}

{self.failed_test_case}
{add_block(test_feedback)}

{self.explanation_and_correct_program}
'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']

        print_log('generate code repair [system]', system_prompt, 1)
        print_log('generate code repair [user]', user_prompt, 1)
        print_log('generate code repair [assistant]', output, 1)

        code = extract_code(output)
        print_log('generate code repair [code]', code, 1)

        return {
            'code': code,
            'output': output,
            'tokens_count': gen['tokens_count']
        }

    def generate_reflexion(
            self,
            prompt: str,
            item: Dict,
            history: Dict,
            max_message_tokens: int = 256,
            max_tokens: int = 1024,
            temperature: float = 0.2
    ) -> Dict[str, Any]:
        assert item is not None
        assert item['score'] < 1.0

        code = item['code']
        test_feedback = item['feedbacks'][0]['message']

        if history is not None:
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
The user has made some modifications, but the program is still wrong.
Your task is to explain in natural language why this program is still wrong and suggest changes. Do not write any code.'''

            history_code = history['code']
            history_test_feedback = history['feedbacks'][0]['message']

            reflection_message = item['reflection_message']

            user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.incorrect_program}
{add_block(history_code)}

{self.failed_test_case}
{history_test_feedback}

{self.previous_explanation_and_modification}
{reflection_message}
{add_block(code)}

{self.failed_test_case_of_previous_modification}
{test_feedback}

{self.reflection_message}
'''
        else:
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
Your task is to explain in natural language why this program is wrong and suggest changes. Do not write any code.'''

            user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.incorrect_program}
{add_block(code)}

{self.failed_test_case}
{test_feedback}

{self.reflection_message}
'''

        gen = self.model.generate_chat(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            max_tokens=max_message_tokens,
            temperature=temperature
        )
        reflection_message = gen['output']

        print_log('generate reflection message [system]', system_prompt, 1)
        print_log('generate reflection message [user]', user_prompt, 1)
        print_log('generate reflection message [assistant]', reflection_message, 1)


        if history is not None:
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
The user has made some modifications, but the program is still wrong.
Your task is to write the correct program according to the reflection message.
Write the program in a Python code block, and do not including docstring in it.'''
        else:
            system_prompt = '''\
You are an expert Python programmer.
You will be provided with a function signature and its docstring, along with an incorrect function implementation program and a failed test case.
Your task is to write the correct program according to the reflection message.
Write the program in a Python code block, and do not including docstring in it.'''

        user_prompt += reflection_message

        gen = self.model.generate_chat(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        output = gen['output']
        code = extract_code(output)

        print_log('generate reflexion [system]', system_prompt, 1)
        print_log('generate reflexion [user]', user_prompt, 1)
        print_log('generate reflexion [assistant]', output, 1)
        print_log('generate reflexion [code]', code, 1)

        return {
            'code': code,
            'reflection_message': reflection_message
        }
