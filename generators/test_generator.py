from utils import extract_test_cases, print_log, add_block
from typing import List, Dict, Any
from code_models import ModelBase
import random


class TestGenerator:
    function_signature_and_docstring = '### Function Signature and Docstring'
    test_cases = '### Test Cases'

    existing_test_cases = '### Existing Test Cases'
    additional_test_cases = '### Additional Test Cases'

    explanation_and_additional_test_cases = '### Explanation and Additional Test Cases'

    compressed_test_cases = '### Compressed Test Cases'

    program_under_testing_and_coverage = '### Program Under Testing and Coverage'

    def __init__(
            self,
            model: ModelBase
    ) -> None:
        self.model = model

    def generate(
            self,
            prompt: str,
            entry_point: str,
            generate_mode: str = 'sample',
            existing_tests: List[str] = [],
            max_tests_per_generation: int = 10,
            max_feedback_tests: int = 10,
            program_feedback: str = '',
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        if generate_mode == 'sample':
            tests = self.generate_test_cases(
                prompt=prompt,
                entry_point=entry_point,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'additional':
            assert len(existing_tests) > 0, 'existing tests are required for additional mode'

            tests = self.generate_additional_test_cases(
                prompt=prompt,
                entry_point=entry_point,
                existing_tests=existing_tests,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'additional_with_feedback':
            assert len(existing_tests) > 0, 'existing tests are required for additional_with_feedback mode'
            assert program_feedback != '', 'program feedback is required for additional_with_feedback mode'

            tests = self.generate_additional_test_cases_with_feedback(
                prompt=prompt,
                entry_point=entry_point,
                existing_tests=existing_tests,
                program_feedback=program_feedback,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'offspring':
            assert len(existing_tests) > 0, 'existing tests are required for offspring mode'

            tests = self.generate_offspring_test_cases(
                prompt=prompt,
                entry_point=entry_point,
                existing_tests=existing_tests,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            raise NotImplementedError(f'generate mode {generate_mode} is not implemented')

        if len(existing_tests) > 0:
            tests = list(set(tests) - set(existing_tests))
        tests = tests[: max_tests_per_generation]
        return {
            'tests': tests
        }

    def generate_test_cases(
            self,
            prompt: str,
            entry_point: str,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
You will be provided with a function signature and its docstring.
Your task is to write test cases for the function. Each test case should be represented by a single line of assert statement.
Write the test cases in a Python code block.'''

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.test_cases}
'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate test cases [system]', system_prompt, 1)
        print_log('generate test cases [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate test cases [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate test cases [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    def generate_additional_test_cases(
            self,
            prompt: str,
            entry_point: str,
            existing_tests: List[str],
            max_feedback_tests: int = 10,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
You will be provided with a function signature and its docstring, along with some existing test cases, but these test cases may not be able to distinguish all faulty programs.
Your task is to write some additional test cases for the function, the additional test cases should be diverse and cover edge cases.
Write each test case in a single line of assert statement, and the length of a single test case should not exceed 512 characters.
Write the additional test cases in a Python code block, and do not repeating the existing ones.'''

        existing_tests = list(set(existing_tests))
        existing_tests = [t for t in existing_tests if len(t) <= 1024]
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n'.join(existing_tests).strip() + '\n...'
        else:
            existing_tests = '\n'.join(existing_tests).strip()

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.existing_test_cases}
{add_block(existing_tests)}

{self.additional_test_cases}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate test cases additional [system]', system_prompt, 1)
        print_log('generate test cases additional [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate test cases additional [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate test cases additional [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    def generate_additional_test_cases_with_feedback(
            self,
            prompt: str,
            entry_point: str,
            existing_tests: List[str],
            program_feedback: str,
            max_feedback_tests: int = -1,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
You will be provided with a function signature and its docstring, along with some existing test cases.
Your task is to write some additional test cases for the function.
You will also be provided with a possible program implementation and the coverage of the existing tests on it, where each line starts with `[+]` to indicate that the line is covered, and `[-]` to indicate that it is not covered.
If there are uncovered lines, please try to write test cases that cover those lines. If all lines are covered, please try to write stronger test cases to check whether the program is correct.
Write each test case in a single line of assert statement, and the length of a single test case should not exceed 512 characters.
Write the additional test cases in a Python code block, and do not repeating the existing ones.'''

        # remove duplicated
        existing_tests = list(set(existing_tests))
        existing_tests = [t for t in existing_tests if len(t) <= 1024]
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n'.join(existing_tests).strip() + '\n...'
        else:
            existing_tests = '\n'.join(existing_tests).strip()

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.existing_test_cases}
{add_block(existing_tests)}

{self.program_under_testing_and_coverage}
{add_block(program_feedback)}

{self.additional_test_cases}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate additional test cases with feedback [system]', system_prompt, 1)
        print_log('generate additional test cases with feedback [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate additional test cases with feedback [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate additional test cases with feedback [test cases]', '\n'.join(test_cases), 1)

        return test_cases





    def generate_population_init(
            self,
            prompt: str,
            entry_point: str,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
Your task is to write test cases for a given function based on its signature and docstring.
 - Each test case should include one line of assert statement, do not define classes or functions.
 - Add a line of brief comment before each test case explaining its purpose.
 - Do not numbering the test cases.
Provide the test cases in a Python code block.'''

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.test_cases}
'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate init test cases [system]', system_prompt, 1)
        print_log('generate init test cases [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate init test cases [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate init test cases [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    pre_written_test_cases = '### Pre-Written Test Cases'
    description_and_additional_test_cases = '### Description and Additional Test Cases'

    def generate_population_offspring(
            self,
            prompt: str,
            entry_point: str,
            existing_tests: List[str],
            max_feedback_tests: int = -1,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
You will receive a function signature, its docstring, and some pre-written test cases.
Your primary task is to rethink the test coverage thoroughly and identify any missing cases to ensure comprehensive testing.

Instructions:

1. Start with a brief description (2-3 sentences) of which cases are missing and why they matter.
2. Write the additional test cases in a Python code block.
 - Each test case should be a single line of assert statement.
 - Add a short comment before each test case to explain its purpose.
 - At least write 10 additional test cases, and ensure that the length of each test case is less than 512 characters.
3. Do not repeat the existing test cases.'''

        # remove duplicated
        existing_tests = list(set(existing_tests))
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n\n'.join(existing_tests).strip() + '\n\n...'
        else:
            existing_tests = '\n\n'.join(existing_tests).strip()

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.pre_written_test_cases}
{add_block(existing_tests)}

{self.description_and_additional_test_cases}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate offspring test cases [system]', system_prompt, 1)
        print_log('generate offspring test cases [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate offspring test cases [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate offspring test cases [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    example_program_and_coverage = '### Example Program and Coverage'
    def generate_population_offspring_with_feedback(
            self,
            prompt: str,
            entry_point: str,
            existing_tests: List[str],
            program_feedback: str,
            max_feedback_tests: int = -1,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
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

        # remove duplicated
        existing_tests = list(set(existing_tests))
        # existing_tests = [t for t in existing_tests if len(t) <= 1024]
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n\n'.join(existing_tests).strip() + '\n\n...'
        else:
            existing_tests = '\n\n'.join(existing_tests).strip()

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.pre_written_test_cases}
{add_block(existing_tests)}

{self.example_program_and_coverage}
{add_block(program_feedback)}

{self.description_and_additional_test_cases}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate offspring test cases [system]', system_prompt, 1)
        print_log('generate offspring test cases [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate offspring test cases [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate offspring test cases [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    def generate_population(
            self,
            prompt: str,
            entry_point: str,
            generate_mode: str = 'init',
            existing_tests: List[str] = [],
            max_tests_per_generation: int = 10,
            max_feedback_tests: int = 10,
            program_feedback: str = '',
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        if generate_mode == 'init':
            tests = self.generate_population_init(
                prompt=prompt,
                entry_point=entry_point,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'offspring':
            assert len(existing_tests) > 0, 'existing tests are required for offspring mode'

            tests = self.generate_population_offspring(
                prompt=prompt,
                entry_point=entry_point,
                existing_tests=existing_tests,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'offspring_with_feedback':
            assert len(existing_tests) > 0, 'existing tests are required for offspring_with_feedback mode'
            assert program_feedback != '', 'program feedback is required for offspring_with_feedback mode'

            tests = self.generate_population_offspring_with_feedback(
                prompt=prompt,
                entry_point=entry_point,
                existing_tests=existing_tests,
                program_feedback=program_feedback,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            raise NotImplementedError(f'generate mode {generate_mode} is not implemented')

        if len(existing_tests) > 0:
            tests = list(set(tests) - set(existing_tests))
        tests = tests[: max_tests_per_generation]
        return {
            'tests': tests
        }

    def generate_offspring_test_cases(
            self,
            prompt: str,
            entry_point: str,
            existing_tests: List[str],
            max_feedback_tests: int = -1,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
You will be provided with a function signature, its docstring, and a set of existing test cases.
Your task is to design additional test cases to thoroughly verify the correctness of the user's implementation program.
The additional test cases you create should address scenarios and edge cases not covered by the existing ones, ensuring that the test suite can effectively identify potential errors in the implementation.
Focus on including edge cases, boundary values, and other inputs that may lead to failure or unexpected behavior.
Make sure the additional test cases are well-documented and align with the function's requirements as described in the provided docstring.
Write each test case in a single line of assert statement, and the length of a single test case should not exceed 512 characters.
Write the additional test cases in a Python code block, and do not repeating the existing ones.'''

        # remove duplicated
        existing_tests = list(set(existing_tests))
        existing_tests = [t for t in existing_tests if len(t) <= 1024]
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n'.join(existing_tests).strip() + '\n...'
        else:
            existing_tests = '\n'.join(existing_tests).strip()

        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.existing_test_cases}
{add_block(existing_tests)}

{self.additional_test_cases}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate offspring test cases [system]', system_prompt, 1)
        print_log('generate offspring test cases [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate offspring test cases [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate offspring test cases [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    def generate_compressed_test_cases(
            self,
            prompt: str,
            entry_point: str,
            existing_tests: List[str],
            max_tokens: int = 1024,
    ) -> List[str]:
        system_prompt = '''\
You are an expert Python test programmer.
You will be provided with a function signature and its docstring, along with some existing test cases.
Your task is to compress these test cases, remove test cases with the same effect, and give a minimal subset.
Write your result in a Python code block.'''

        existing_tests = list(set(existing_tests))
        existing_tests = '\n'.join(existing_tests).strip()
        user_prompt = f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}

{self.existing_test_cases}
{add_block(existing_tests)}

{self.compressed_test_cases}
'''

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate compressed test cases [system]', system_prompt, 1)
        print_log('generate compressed test cases [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2
        )
        output = gen['output']
        print_log('generate compressed test cases [assistant]', output, 1)
        test_cases = extract_test_cases(output, entry_point)
        print_log('generate compressed test cases [test cases]', '\n'.join(test_cases), 1)

        return test_cases

    def exit(self):
        pass
