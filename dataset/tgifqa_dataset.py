import os
import pickle
import random
from logging import Logger

import ipdb
import numpy as np
import torch
from pyhocon import ConfigTree
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, get_worker_info
from typing import List

from utils.npy_file import NpyFile


class VQAFeatureDataset(Dataset):

    def __init__(self, opt: ConfigTree, logger: Logger, split: str):
        self.split = split
        self.logger = logger

        self.task = opt.get_string('task')
        self.num_frames = opt.get_int('num_frames')
        self.question_path = os.path.join(
            opt.get_string('question_path'),
            f'{split.capitalize()}_{self.task}_question.pkl'
        )
        self.feature_path = opt.get_string('feature_path')
        self.use_bbox_features = opt.get_bool('use_bbox_features')
        self.c3d_feature_path = opt.get_string('c3d_feature_path')

        if self.use_bbox_features:
            logger.info('Using bbox features!')
            self.bbox_features_path = opt.get_string('bbox_feature_path')
            self.bbox_features = NpyFile(self.bbox_features_path)
            # self.label_path = opt.get_string('label_path')
            # self.labels = NpyFile(self.label_path)
            # self.score_path = opt.get_string('score_path')
            # self.scores = NpyFile(self.score_path)
            self.bbox_path = opt.get_string('bbox_path')
            self.bbox = NpyFile(self.bbox_path)
            self.num_bbox = opt.get_int('num_bbox')

        if self.c3d_feature_path is not None:
            self.c3d_features = NpyFile(self.c3d_feature_path)

        self.frame_range = np.arange(self.num_frames)
        self.samples = self._load_questions()
        self.features = self._load_feature()

    def __getitem__(self, index: int):
        sample = self.samples[index]

        question = sample['question_ids']
        question_length = len(question)
        answer_id = sample['answer_id']
        gif_name = sample['gif_name']
        question_chars = sample['question_chars']

        features = self.features[gif_name]

        num_frames = features.shape[0]
        if self.c3d_feature_path is not None:
            c3d_features = self.c3d_features[gif_name]
            num_frames = min(c3d_features.shape[0], num_frames)
        else:
            c3d_features = torch.zeros(1)

        feature_indices = self._resample(num_frames)
        features = features[feature_indices]

        if self.c3d_feature_path is not None:
            c3d_features = c3d_features[feature_indices]
            c3d_features = torch.from_numpy(c3d_features)

        if self.task in ['action', 'transition']:
            answers = [torch.LongTensor(answer)
                       for answer in sample['answer_ids']]
            answer_chars = sample['answer_chars']
        else:
            answers = [torch.LongTensor([0]) for _ in range(5)]
            answer_chars = [torch.LongTensor([0]) for _ in range(5)]

        answer_lengths = [len(answer) for answer in answers]

        features = torch.from_numpy(features)

        question = torch.LongTensor(question)

        question_chars = torch.LongTensor(question_chars)

        if self.use_bbox_features:
            bbox_features = self.bbox_features[gif_name][feature_indices, :self.num_bbox]
            bbox = self.bbox[gif_name][feature_indices, :self.num_bbox]
            # scores = self.scores[gif_name][feature_indices, :self.num_bbox]
            # labels = self.labels[gif_name][feature_indices, :self.num_bbox]
              
            bbox_features = torch.from_numpy(bbox_features)
            bbox = torch.from_numpy(bbox)
            # scores = torch.from_numpy(scores)
            # labels = torch.from_numpy(labels)
        else:
            bbox_features = torch.FloatTensor([0])
            bbox = torch.FloatTensor([0])
            # scores = torch.FloatTensor([0])
            # labels = torch.LongTensor([0])

        return (
            question, question_length, question_chars,
            answers[0], answer_lengths[0], answer_chars[0],
            answers[1], answer_lengths[1], answer_chars[1],
            answers[2], answer_lengths[2], answer_chars[2],
            answers[3], answer_lengths[3], answer_chars[3],
            answers[4], answer_lengths[4], answer_chars[4],
            features, c3d_features, bbox_features, bbox,
            answer_id
        )

    def __len__(self):
        return len(self.samples)

    def _resample(self, n: int) -> np.ndarray:
        gap = n / self.num_frames
        new_range = gap * self.frame_range

        if self.split == 'train' and n > self.num_frames:
            new_range += random.random() * gap

        new_range = new_range.astype(np.int64)

        return new_range

    def _load_questions(self):
        self.logger.info(f'loading questions from {self.question_path}')
        with open(self.question_path, 'rb') as f:
            return pickle.load(f)

    def _load_feature(self):
        return NpyFile(self.feature_path)

    def _load_bbox_features(self):
        splits = os.listdir(self.bbox_features_path)


def cat_into_shared_memory(batch: List[torch.Tensor]):
    out = None
    elem = batch[0]
    if get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.stack(batch, 0, out=out)


def collate_fn(batch):
    (
        question, question_length, question_chars,
        a1, a1_length, a1_chars,
        a2, a2_length, a2_chars,
        a3, a3_length, a3_chars,
        a4, a4_length, a4_chars,
        a5, a5_length, a5_chars,
        features, c3d_features, bbox_features, bbox,
        answer
    ) = zip(*batch)

    question = pad_sequence(question, batch_first=True)
    question_length = torch.LongTensor(question_length)
    question_chars = pad_sequence(question_chars, batch_first=True)

    a1 = pad_sequence(a1, batch_first=True)
    a1_length = torch.LongTensor(a1_length)
    a1_chars = pad_sequence(a1_chars, batch_first=True)
    a2 = pad_sequence(a2, batch_first=True)
    a2_length = torch.LongTensor(a2_length)
    a2_chars = pad_sequence(a2_chars, batch_first=True)
    a3 = pad_sequence(a3, batch_first=True)
    a3_length = torch.LongTensor(a3_length)
    a3_chars = pad_sequence(a3_chars, batch_first=True)
    a4 = pad_sequence(a4, batch_first=True)
    a4_length = torch.LongTensor(a4_length)
    a4_chars = pad_sequence(a4_chars, batch_first=True)
    a5 = pad_sequence(a5, batch_first=True)
    a5_length = torch.LongTensor(a5_length)
    a5_chars = pad_sequence(a5_chars, batch_first=True)

    features = cat_into_shared_memory(features)
    c3d_features = cat_into_shared_memory(c3d_features)
    bbox_features = cat_into_shared_memory(bbox_features)
    bbox = cat_into_shared_memory(bbox)
    # scores = cat_into_shared_memory(scores)
    # labels = cat_into_shared_memory(labels)

    answer = torch.tensor(answer)

    return (
        question, question_length, question_chars,
        a1, a1_length, a1_chars,
        a2, a2_length, a2_chars,
        a3, a3_length, a3_chars,
        a4, a4_length, a4_chars,
        a5, a5_length, a5_chars,
        features, c3d_features, bbox_features, bbox,
        answer
    )


def get_dataset(opt: ConfigTree, logger: Logger) -> (Dataset, Dataset):
    train_set = VQAFeatureDataset(opt, logger, 'train')

    test_set = VQAFeatureDataset(opt, logger, 'test')

    return train_set, test_set
