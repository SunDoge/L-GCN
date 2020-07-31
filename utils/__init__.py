import pandas as pd
from .dictionary import Dictionary, CharDictionary
import pickle
from typing import Dict, List
import torch
import os
import numpy as np
# from torchpie.config import config

MULTIPLE_CHOICE_TASKS = ['action', 'transition']


def load_csv_from_dataset(task: str, split: str) -> pd.DataFrame:
    return pd.read_csv(f'data/dataset/{split}_{task}_question.csv', sep='\t')


def load_dictionary(config, task: str, level: str) -> Dictionary:
    cache_path = config.get_string('cache_path')

    if level == 'word':
        return Dictionary.load_from_file(f'{cache_path}/{task}_dictionary.pkl')
    elif level == 'char':
        return CharDictionary.load_from_file(f'{cache_path}/{task}_char_dictionary.pkl')


def load_answer_dict() -> Dict[str, int]:
    with open('cache/frameqa_answer_dict.pkl', 'rb') as f:
        return pickle.load(f)


def get_vocab_size(config, task: str, level: str = 'word') -> int:
    dictionary = load_dictionary(config, task, level)
    return len(dictionary.idx2word)


def batch_to_gpu(batch: List[torch.Tensor]) -> List[torch.Tensor]:
    new_batch = [x.cuda(non_blocking=True) for x in batch]
    return new_batch


def load_pretrained_character_embedding(path='data/glove.840B.300d-char.txt') -> (Dict[str, int], torch.Tensor):
    cached_path = os.path.join('.vector_cache', 'glove.840B.300d-char.txt.pt')
    if os.path.exists(cached_path):
        print('Cache exists')
        return torch.load(cached_path)
    char2index = dict()
    data = []
    with open(path, 'r') as f:
        for index, line in enumerate(f.readlines()):
            line_split = line.strip().split()
            vec = np.array(line_split[1:], dtype=np.float32)
            char = line_split[0]
            char2index[char] = index
            data.append(vec)

    data = np.array(data)
    data = torch.from_numpy(data)

    print('Build cache')
    torch.save([char2index, data], cached_path)

    return char2index, data


@torch.no_grad()
def count_correct(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_id = pred.argmax(dim=1)
    corrects = pred_id.eq(target).sum()
    return corrects


@torch.no_grad()
def accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_corrects = count_correct(pred, target)
    acc = num_corrects.float() * 100. / target.shape[0]
    return acc
