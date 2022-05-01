from __future__ import annotations

import gc
import subprocess
from typing import Tuple, Iterable

import torch


def get_gpu_stats() -> Tuple[int, int, int]:
    """
    ref: https://github.com/huggingface/transformers/issues/1742
    """

    def query(field):
        return (subprocess.check_output(
            [
                'nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = 100 * used / total

    return used, total, pct


def get_gpu_status_str() -> str:
    used, total, pct = get_gpu_stats()
    return f"{pct:2.1f}% ({used} out of {total})"


def get_tracked_tensors() -> Iterable[torch.Tensor]:
    for tracked_object in gc.get_objects():
        if torch.is_tensor(tracked_object):
            yield tracked_object


def get_tracked_tensors_strs() -> Iterable[str]:
    for tracked_tensor in get_tracked_tensors():
        yield f"{type(tracked_tensor).__name__} " \
              f"{tracked_tensor.shape} " \
              f"{'GPU ' if tracked_tensor.is_cuda else ''}" \
              f"{'pinned ' if tracked_tensor.is_pinned() else ''}"


def cleanup(func):

    def inner(*args, **kwargs):
        results = func(*args, **kwargs)

        gc.collect()
        torch.cuda.empty_cache()

        return results

    return inner


def get_cuda_device_stats(device=None):
    assert torch.cuda.is_available()
    if device is None:
        device = torch.cuda.current_device()

    return torch.cuda.get_device_name(device), torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)


def get_cuda_device_stats_str(device=None):
    device_name, memory_allocated, memory_reserved = get_cuda_device_stats(device)

    return f"{device_name} (Allocated: {memory_allocated} | Reserved: {memory_reserved})"
