import json
from typing import List, Dict, Union
import os


def read_jsonl(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        return []
    data = []
    with open(file_path, 'r') as file:
        line = file.readline()
        while line != '':
            line = json.loads(line)
            data.append(line)
            line = file.readline()
    return data


def write_jsonl(file_path: str, data: Union[List[Dict], Dict]):
    if type(data) == dict:
        data = [data]
    with open(file_path, 'w') as file:
        for line in data:
            line = json.dumps(line)
            file.write(line + '\n')


def append_jsonl(file_path: str, data: Union[List[Dict], Dict]):
    if type(data) == dict:
        data = [data]
    with open(file_path, 'a') as file:
        for line in data:
            line = json.dumps(line)
            file.write(line + '\n')


def dir_jsonl_files(dir_path: str) -> List[str]:
    i = 0
    res = []
    while True:
        file_path = os.path.join(dir_path, f'result_{i}.jsonl')
        if not os.path.exists(file_path):
            break
        res.append(file_path)
        i += 1
    return res
