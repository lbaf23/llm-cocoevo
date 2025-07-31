from utils import extract_test_cases, extract_unit_tests, print_log, add_block
from typing import List, Dict, Union, Any
from code_models import ModelBase
import random
from .utils import get_test_system_prompt


def extract_tests(content: str, env_type: str, entry_point: str = '') -> List[str]:
    if env_type == 'func':
        return extract_test_cases(content, entry_point)
    elif env_type == 'real_world_function':
        return extract_unit_tests(content)
    else:
        raise ValueError('Invalid env type')

class TestGenerator:
    function_signature_and_docstring = '### Function Signature and Docstring'
    test_cases = '### Test Cases'

    # for real world
    program_skeleton = '### Program Skeleton'
    test_prefix = '### Test Prefix'
    unit_tests_inst = '''\
Write unit tests for the function with a `# TODO` sign. The test prefix are some requirements that you may rely on, do not repeat this part in your result. Write your result in a Python code block, for example:
```python
def test_xxx():
    # ...

def test_xxx():
    # ...

...
```
'''

    example_program_and_coverage = '### Example Program and Coverage'

    pre_written_test_cases = '### Pre-Written Test Cases'

    description_and_additional_test_cases = '### Description and Additional Test Cases'

    def __init__(
            self,
            model: ModelBase
    ) -> None:
        self.model = model

    def make_prefix_prompt(self, env_type: str, prompt: str, data_args: Dict[str, Any]) -> str:
        if env_type == 'func':
            return f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}'''
        elif env_type == 'real_world_function':
            return f'''\
{self.unit_tests_inst}

{self.program_skeleton}
{add_block(prompt)}

{self.test_prefix}
{add_block(data_args['prompt_test'])}
'''
        else:
            raise NotImplementedError

    def generate(
            self,
            prompt: str,
            entry_point: str,
            env_type: str,
            data_args: Dict[str, Any],
            existing_tests: List[str] = None,
            max_tests_per_generation: int = 10,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        existing_tests = [] if existing_tests is None else existing_tests
        gen = self.generate_test_cases(
            prompt=prompt,
            env_type=env_type,
            data_args=data_args,
            entry_point=entry_point,
            max_tokens=max_tokens,
            temperature=temperature
        )
        tests = gen['tests']
        if len(existing_tests) > 0:
            tests = list(set(tests) - set(existing_tests))
        tests = tests[: max_tests_per_generation]
        return {
            'tests': tests,
            'tokens_count': gen['tokens_count']
        }

    def generate_population(
            self,
            prompt: str,
            entry_point: str,
            generate_mode: str,
            env_type: str,
            data_args: Dict[str, Any],
            existing_tests: List[str] = None,
            max_tests_per_generation: int = 10,
            max_feedback_tests: int = 10,
            program_feedback: str = '',
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        """

        Args:
            prompt:
            entry_point:
            generate_mode (str):
                random: generate random test cases
                population: generate additional test cases
                population_and_feedback: generate additional test cases with coverage feedback
                feedback: generate test cases with coverage feedback
            existing_tests (List[str]):
            max_tests_per_generation:
            max_feedback_tests:
            program_feedback:
            max_tokens:
            temperature:

        Returns:
            (Dict[str, Any]):
                tests (List[str]):
                tokens_count (Dict[str, int]):


        """

        existing_tests = [] if existing_tests is None else existing_tests
        if generate_mode == 'random':
            gen = self.generate_test_cases(
                prompt=prompt,
                env_type=env_type,
                data_args=data_args,
                entry_point=entry_point,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'population' or (generate_mode == 'population_and_feedback' and program_feedback == ''):
            assert len(existing_tests) > 0, 'existing tests are required for offspring mode'

            if (generate_mode == 'population_and_feedback' and program_feedback == ''):
                print_log('Warning: program_feedback is empty, fall back to population mode.', '')

            gen = self.generate_test_cases_with_population(
                prompt=prompt,
                env_type=env_type,
                data_args=data_args,
                entry_point=entry_point,
                existing_tests=existing_tests,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'population_and_feedback':
            assert len(existing_tests) > 0, 'existing tests are required for offspring_with_feedback mode'
            assert program_feedback != '', 'program feedback is required for offspring_with_feedback mode'

            gen = self.generate_test_cases_with_population_and_feedback(
                prompt=prompt,
                entry_point=entry_point,
                env_type=env_type,
                data_args=data_args,
                existing_tests=existing_tests,
                program_feedback=program_feedback,
                max_feedback_tests=max_feedback_tests,
                max_tokens=max_tokens,
                temperature=temperature
            )
        elif generate_mode == 'feedback':
            assert program_feedback != '', 'program feedback is required for offspring_with_feedback mode'

            gen = self.generate_test_cases_with_feedback(
                prompt=prompt,
                entry_point=entry_point,
                env_type=env_type,
                data_args=data_args,
                program_feedback=program_feedback,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            raise NotImplementedError(f'generate mode {generate_mode} is not implemented')

        tests = gen['tests']
        if len(existing_tests) > 0:
            tests = list(set(tests) - set(existing_tests))
        tests = tests[: max_tests_per_generation]
        return {
            'tests': tests,
            'tokens_count': gen['tokens_count']
        }

    def generate_test_cases(
            self,
            prompt: str,
            entry_point: str,
            env_type: str,
            data_args: Dict[str, Any],
            max_tokens: int = 1024,
            temperature: float = 0.8,
    ) -> Dict[str, Any]:
        system_prompt = get_test_system_prompt('generation', env_type)
        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
        test_cases = extract_tests(output, env_type, entry_point)
        print_log('generate init test cases [test cases]', '\n\n'.join(test_cases), 1)

        return {
            'tests': test_cases,
            'tokens_count': gen['tokens_count']
        }

    def generate_test_cases_with_population(
            self,
            prompt: str,
            entry_point: str,
            env_type: str,
            data_args: Dict[str, Any],
            existing_tests: List[str],  # test population
            max_feedback_tests: int = -1,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        system_prompt = get_test_system_prompt('population', env_type)
        # remove duplicated
        existing_tests = list(set(existing_tests))
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n\n'.join(existing_tests).strip() + '\n\n...'
        else:
            existing_tests = '\n\n'.join(existing_tests).strip()

        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
        test_cases = extract_tests(output, env_type, entry_point)
        print_log('generate offspring test cases [test cases]', '\n\n'.join(test_cases), 1)

        return {
            'tests': test_cases,
            'tokens_count': gen['tokens_count']
        }

    def generate_test_cases_with_population_and_feedback(
            self,
            prompt: str,
            entry_point: str,
            env_type: str,
            data_args: Dict[str, Any],
            existing_tests: List[str],
            program_feedback: str,
            max_feedback_tests: int = -1,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        system_prompt = get_test_system_prompt('population_and_feedback', env_type)
        # remove duplicated
        existing_tests = list(set(existing_tests))
        if max_feedback_tests > 0 and len(existing_tests) > max_feedback_tests:
            existing_tests = random.sample(existing_tests, max_feedback_tests)
            existing_tests = '\n\n'.join(existing_tests).strip() + '\n\n...'
        else:
            existing_tests = '\n\n'.join(existing_tests).strip()

        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
        test_cases = extract_tests(output, env_type, entry_point)
        print_log('generate offspring test cases [test cases]', '\n\n'.join(test_cases), 1)

        return {
            'tests': test_cases,
            'tokens_count': gen['tokens_count']
        }

    def generate_test_cases_with_feedback(
            self,
            prompt: str,
            entry_point: str,
            env_type: str,
            data_args: Dict[str, Any],
            program_feedback: str,
            max_tokens: int = 1024,
            temperature: float = 0.8
    ) -> Dict[str, Any]:
        system_prompt = get_test_system_prompt('feedback', env_type)
        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

{self.example_program_and_coverage}
{add_block(program_feedback)}

{self.description_and_additional_test_cases}
'''
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

        print_log('generate_test_cases_with_feedback [system]', system_prompt, 1)
        print_log('generate_test_cases_with_feedback [user]', user_prompt, 1)

        gen = self.model.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        output = gen['output']
        print_log('generate_test_cases_with_feedback [assistant]', output, 1)
        test_cases = extract_tests(output, env_type, entry_point)
        print_log('generate_test_cases_with_feedback [test cases]', '\n\n'.join(test_cases), 1)

        return {
            'tests': test_cases,
            'tokens_count': gen['tokens_count']
        }

