import ast
from typing import List, Any, Dict
import black
import re


def extract_first_block(output: str):
    output = output.split('\n')

    i = 0
    while i < len(output):
        if output[i].strip().startswith('```'):
            break
        i += 1
    i += 1
    res = ''
    while i < len(output):
        if output[i].strip().startswith('```'):
            break
        res += output[i] + '\n'
        i += 1
    return res.strip()


def is_syntax_valid(code: str, lang: str = 'python') -> bool:
    if lang == 'python':
        try:
            ast.parse(code)
            return True
        except Exception:
            return False
    else:
        raise NotImplementedError


def extract_blocks(content: str) -> List[str]:
    """
    extract all blocks content and return as a list
    """
    blocks = []
    if content.__contains__('```'):
        lines = content.strip().split('\n')
        i = 0
        block = ''
        while i < len(lines):
            while i < len(lines):
                if lines[i].strip().startswith('```'):
                    i += 1
                    break
                i += 1
            block = ''
            while i < len(lines):
                if lines[i].strip().startswith('```'):
                    i += 1
                    blocks.append(block.strip())
                    block = ''
                    break
                block += lines[i] + '\n'
                i += 1
        if block != '':
            blocks.append(block)
    return blocks


def format_test_with_comment(test: str) -> str:
    lines = test.split('\n')
    lines = [l for l in lines if l.strip() != '']
    i = 0
    while i < len(lines):
        if lines[i].__contains__('assert'):
            break
        i += 1

    if i > 0:
        lines = lines[i - 1 : ]
    if i < len(lines) - 1:
        lines = lines[ : i + 1]
    return '\n'.join(lines)



def filter_tests(tests: List[str]) -> List[str]:
    res_tests = []
    for test in tests:
        t = ''
        lines = test.split('\n')
        for l in lines:
            t += l + '\n'
            if l.startswith('assert'):
                break
        try:
            t = format_code(t, mode='soft')
            if not res_tests.__contains__(t):
                t = format_test_with_comment(t)
                res_tests.append(t)
        except Exception:
            pass
    return res_tests


def extract_test_cases(output: str, entry_point: str) -> List[str]:
    blocks = extract_blocks(output)
    for i in range(0, len(blocks)):
        try:
            blocks[i] = format_code(blocks[i])
        except Exception:
            pass
    
    if len(blocks) > 0:
        content = '\n'.join(blocks)
    else:
        content = output

    tests = []

    lines = content.split('\n')
    i = 0
    prefix = ''

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('assert'):
            tests.append(prefix + line)
            prefix = ''
        else:
            if line != '':
                prefix += line + '\n'
        i += 1

    tests = [t for t in tests if t.__contains__(entry_point)]
    return tests


def extract_code(content: str, lang: str = 'python') -> str:
    """
    extract a program
    """
    code = content
    blocks = extract_blocks(content)
    if len(blocks) > 0:
        code = blocks[0]
    return code.strip()


def add_block(content: str, lang: str = 'python') -> str:
    return f'''\
```{lang.strip()}
{content.strip()}
```'''


# filter text
def filter_long_text(text: Any, max_length: int = 512, suffix_length: int = 32) -> str:
    """
    123456789123456789123456789  -->  123 ... 89
    """
    text = str(text)
    if len(text) > max_length:
        res = text[: max_length - suffix_length - 3] + '...'
        if suffix_length > 0:
            res += text[-suffix_length : ]
        return res
    else:
        return text


def filter_test_case_text(test_case: str, length: int = 128) -> str:
    lines = test_case.strip().split('\n')
    test_case = ''
    for line in lines:
        if line.__contains__('=='):
            line = filter_long_text(line[ : line.index('==')].strip(), length, 0) + ' == ' + filter_long_text(line[line.index('==') + 2 : ].strip(), length, 0)
        else:
            line = filter_long_text(line, length, 0)
        test_case += line + '\n'
    return test_case.strip()


import sys
from io import StringIO
import os


class StdUtils:
    def __init__(self) -> None:
        self.redirect_pipe = False
        self.redirect_str = False

        self.stdin = sys.stdin
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def redirect_pipe_io(
            self,
            input_content: str = ''
    ):
        self.redirect_pipe = True

        # in
        self.input_read_fd, self.input_write_fd = os.pipe()
        self.input_read_file = os.fdopen(self.input_read_fd, 'r')
        self.input_write_file = os.fdopen(self.input_write_fd, 'w')

        # write input_content
        self.input_write_file.write(input_content)
        self.input_write_file.close()  # add EOF

        # err
        self.err_read_fd, self.err_write_fd = os.pipe()
        self.err_read_file = os.fdopen(self.err_read_fd, 'r')
        self.err_write_file = os.fdopen(self.err_write_fd, 'w')

        # out
        self.print_read_fd, self.print_write_fd = os.pipe()
        self.print_read_file = os.fdopen(self.print_read_fd, 'r')
        self.print_write_file = os.fdopen(self.print_write_fd, 'w')

        self.redirect(self.input_read_file, self.print_write_file, self.err_write_file)

    def get_pipe_io_output(self) -> str:
        if not self.print_write_file.closed:
            self.print_write_file.close()
        output = self.print_read_file.read()
        return output

    def redirect(self, io_in = None, io_out = None, io_err = None):
        if io_in != None:
            sys.stdin = io_in
        if io_out != None:
            sys.stdout = io_out
        if io_err != None:
            sys.stderr = io_err

    def recover(self):
        self.close()

        sys.stdin = self.stdin
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def redirect_str_io(
            self,
            input_content: str = ''
    ):
        self.redirect_str = True

        self.str_io_in = StringIO(input_content)
        self.str_io_out = StringIO()
        self.str_io_err = StringIO()
        self.redirect(self.str_io_in, self.str_io_out, self.str_io_err)

    def get_str_io_output(self) -> str:
        return self.str_io_out.getvalue()
    
    def get_str_io_err(self) -> str:
        return self.str_io_err.getvalue()

    def close(self):
        if self.redirect_str:
            self.redirect_str = False
            if not self.str_io_in.closed:
                self.str_io_in.close()
            if not self.str_io_out.closed:
                self.str_io_out.close()
            if not self.str_io_err.closed:
                self.str_io_err.close()
        
        if self.redirect_pipe:
            self.redirect_pipe = False

            if not self.input_read_file.closed:
                self.input_read_file.close()
            if not self.input_write_file.closed:
                self.input_write_file.close()
            
            if not self.err_read_file.closed:
                self.err_read_file.close()
            if not self.err_write_file.closed:
                self.err_write_file.close()
            
            if not self.print_read_file.closed:
                self.print_read_file.close()
            if not self.print_write_file.closed:
                self.print_write_file.close()


def try_format_code(code: str, lang: str = 'python', mode: str = 'soft') -> str:
    try:
        code = format_code(code, lang, mode)
    except Exception as e:
        pass
    return code


def remove_comments(code_str: str) -> str:
    lines = code_str.split('\n')
    i = 0
    code = ''
    while i < len(lines):
        if lines[i].strip() == '"""':
            i += 1
            while lines[i].strip() != '"""' and i < len(lines):
                i += 1
            if i >= len(lines):
                break
            i += 1
        code += lines[i] + '\n'
        i += 1
    return code


def format_code(code: str, lang: str = 'python', mode: str = 'soft') -> str:
    if lang == 'python':
        if mode == 'hard':
            code = ast.unparse(ast.parse(code))
        code = black.format_str(code, mode=black.Mode(line_length=100000))
        code = code.strip()
        return code
    else:
        raise NotImplementedError


def select_unique(lst: List[str]) -> List[str]:
    s = set()
    res = []
    for l in lst:
        if not s.__contains__(l):
            res.append(l)
            s.add(l)
    return res


def get_first_feedback(feedbacks: List[Dict]) -> str:
    messages = [f['message'] for f in feedbacks if f.__contains__('message')]
    if len(messages) > 0:
        return messages[0]
    else:
        return ''


def extract_unit_tests(output: str) -> List[str]:
    code = extract_first_block(output)
    pattern = r'def\s+test_\w+\s*\(.*?\)\s*:.*?(?=\ndef|\Z)'
    matches = re.findall(pattern, code, flags=re.DOTALL)
    return [match.strip() for match in matches]
