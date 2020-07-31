import math
import os
import pickle
from functools import lru_cache

import numpy as np
from numpy.lib.format import open_memmap
from typing import Tuple, List, Dict


class NpyFile:

    def __init__(self, name: str, mode='r'):
        self.name = name
        self.index_path = os.path.join(name, 'indices.pkl')
        self.data_path = os.path.join(name, 'data.npy')

        with open(self.index_path, 'rb') as f:
            self.indices = pickle.load(f)

        self.data = np.load(self.data_path, mmap_mode=mode)

    def __getitem__(self, name: str) -> np.memmap:

        index, length = self.indices[name]

        return self.data[index: index + length]

    def __len__(self):
        return len(self.indices)


class NpyFileBuilder:

    def __init__(self, name: str, shape: Tuple[int], dtype=np.float32, exist_ok: bool = False):
        os.makedirs(name, exist_ok=exist_ok)

        self.name = name
        self.index = 0
        self.fp = open_memmap(
            name,
            mode='w+',
            shape=shape,
            dtype=dtype
        )
        self.indices = dict()

    def insert(self, key: str, value: np.ndarray):
        length = len(value)

        self.indices[key] = [self.index, length]

        self.fp[self.index: self.index + length] = value

        self.index += length


def compress_symmetric(data: np.ndarray) -> np.ndarray:
    assert np.allclose(data, data.T)
    flat = data[np.triu_indices(data.shape[0])]
    return flat


def get_dim_from_length(flat: np.ndarray) -> int:
    '''
    (sqrt(1 + 8 * L) - 1) / 2
    '''
    @lru_cache(maxsize=256)
    def impl(length: int) -> int:
        return int(math.sqrt(length * 8 + 1) - 1) // 2

    return impl(len(flat))


def decompress_symmetric(flat: np.ndarray) -> np.ndarray:
    dim = get_dim_from_length(flat)
    data: np.ndarray = np.zeros((dim, dim), dtype=flat.dtype)
    data[np.triu_indices(dim)] = flat
    data = data + data.T - np.diag(data.diagonal())
    return data
