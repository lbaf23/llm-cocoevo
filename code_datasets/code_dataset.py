import json
from typing import Dict, List, Tuple


class CodeDataset:
    data_range: List[int] = []

    def __init__(
            self,
            name: str,
            dataset_file: str,
            selected_ids: List[int] = [],
            start: int = 0,
            end: int = -1
    ):
        """
        dataset_file: dataset jsonl file path
        start: start index
        end: end index
        """
        self.name = name
        print(f'-------------------- load dataset {dataset_file} [{start}, {end}] --------------------', flush=True)

        self.dataset_file = dataset_file
        self.data = []

        with open(dataset_file, 'r') as file:
            line = file.readline()
            while line != '':
                self.data.append(json.loads(line))
                line = file.readline()

        self.start = start
        self.end = len(self.data) if end == -1 else end
        self.data_range = [i for i in range(self.start, self.end)]
        if len(selected_ids) > 0:
            self.data_range = [i for i in self.data_range if selected_ids.__contains__(i)]

    def get_data(self, i: int) -> Dict:
        """
        i: data index
        return
            data: dataset data i
        """
        if i < self.start or i >= self.end:
            raise IndexError

        d = self.data[i - self.start]

        if self.name == 'leetcode':
            return {
                **d,
                'data_args': {}
            }
        elif self.name == 'real_world_function':
            return {
                'index': d['index'],
                'prompt': d['prompt'],
                'entry_point': d['entry_point'],
                'solution': d['solution'],
                'tests': d['test'],
                'data_args': {
                    'prompt_test': d['prompt_test'],
                    'context_program': d['context_program'],
                    'context_test_program': d['context_test_program'],
                }
            }
        elif self.name == 'real_world_method':
            return {
                'index': d['index'],
                'prompt': d['prompt_code'],
                'entry_point': d['entry_point'],
                'solution': d['solution'],
                'tests': d['tests'],
                'data_args': {
                    'entry_point': d['entry_point'],
                    'program': d['program'],
                    'start_line': d['start_line'],
                    'end_line': d['end_line'],
                    'test_prefix': d['test_prefix']
                }
            }
        else:
            raise NotImplementedError(f'self.name={self.name}')
