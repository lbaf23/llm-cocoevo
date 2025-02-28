import torch
import os
from argparse import ArgumentParser
from typing import Dict, List


def init_cuda():
    if torch.cuda.is_available:
        try:
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
        except Exception:
            print('cuda emtpy cache failed.', flush=True)
