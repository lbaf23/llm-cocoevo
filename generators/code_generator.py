from code_models import ModelBase
from utils import extract_code, print_log, add_block, get_first_feedback
from typing import Dict, Any
from .utils import get_code_system_prompt


class CodeGenerator:
    function_signature_and_docstring = '### Function Signature and Docstring'
    program = '### Program'

    # for real world
    program_skeleton = '### Program Skeleton'
    rw_function_program_skeleton_inst = '''\
Implement the function with a `# TODO` sign. Only write the target function that needs to be implemented, do not repeat the rest part of programs and docstrings.'''

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

    def make_prefix_prompt(self, env_type: str, prompt: str, data_args: Dict[str, Any]) -> str:
        if env_type == 'func':
            return f'''\
{self.function_signature_and_docstring}
{add_block(prompt)}'''
        elif env_type == 'real_world_function':
            return f'''\
{self.rw_function_program_skeleton_inst}

{self.program_skeleton}
{add_block(prompt)}
'''
        else:
            raise NotImplementedError

    def generate(
            self,
            prompt: str,
            env_type: str,
            data_args: Dict[str, Any],
            init_method: str = 'random_prompt',
            max_tokens: int = 1024,
            temperature: float = 0.8,
    ) -> Dict:
        system_prompt = get_code_system_prompt('generation', env_type, init_method)

        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + '\n\n' + self.program + '\n'

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
            env_type: str,
            data_args: Dict[str, Any],
            max_tokens: int = 1024,
            temperature: float = 0.8,
    ) -> Dict[str, Any]:
        system_prompt = get_code_system_prompt('crossover', env_type)
        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
            env_type: str,
            data_args: Dict[str, Any],
            max_tokens: int = 1024,
            temperature: float = 0.8,
    ) -> Dict[str, Any]:
        system_prompt = get_code_system_prompt('mutation', env_type)
        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
            env_type: str,
            data_args: Dict[str, Any],
            test_feedback: str,
            max_tokens: int = 1024,
            temperature: float = 0.8,
    ) -> Dict[str, Any]:
        assert test_feedback != ''

        system_prompt = get_code_system_prompt('repair', env_type)
        user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
            env_type: str,
            data_args: Dict[str, Any],
            max_message_tokens: int = 256,
            max_tokens: int = 1024,
            temperature: float = 0.2,
    ) -> Dict[str, Any]:
        assert item is not None
        assert item['score'] < 1.0

        code = item['code']
        test_feedback = get_first_feedback(item['feedbacks'])

        tokens_count = {
            'prompt_tokens': 0,
            'completion_tokens': 0
        }
        if history is not None:
            history_code = history['code']
            history_test_feedback = get_first_feedback(history['feedbacks'])
            reflection_message = item['reflection_message']

            system_prompt = get_code_system_prompt('reflection', env_type, 'long')

            user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
            system_prompt = get_code_system_prompt('reflection', env_type, 'short')

            user_prompt = self.make_prefix_prompt(env_type, prompt, data_args) + f'''

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
        tokens_count['prompt_tokens'] += gen['tokens_count']['prompt_tokens']
        tokens_count['completion_tokens'] += gen['tokens_count']['completion_tokens']

        reflection_message = gen['output']

        print_log('generate reflection message [system]', system_prompt, 1)
        print_log('generate reflection message [user]', user_prompt, 1)
        print_log('generate reflection message [assistant]', reflection_message, 1)

        if history is not None:
            system_prompt = get_code_system_prompt('reflection_code', env_type, 'long')
        else:
            system_prompt = get_code_system_prompt('reflection_code', env_type, 'short')

        user_prompt += reflection_message

        gen = self.model.generate_chat(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        tokens_count['prompt_tokens'] += gen['tokens_count']['prompt_tokens']
        tokens_count['completion_tokens'] += gen['tokens_count']['completion_tokens']

        output = gen['output']
        code = extract_code(output)

        print_log('generate reflexion [system]', system_prompt, 1)
        print_log('generate reflexion [user]', user_prompt, 1)
        print_log('generate reflexion [assistant]', output, 1)
        print_log('generate reflexion [code]', code, 1)

        return {
            'code': code,
            'reflection_message': reflection_message,
            'tokens_count': tokens_count
        }
