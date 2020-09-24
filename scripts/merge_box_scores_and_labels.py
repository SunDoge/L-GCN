import torch
from typed_args import TypedArgs, add_argument
import argparse
import os
from typing import Dict, List, NewType, Optional
from tqdm import tqdm

from numpy.lib.format import open_memmap
import numpy as np
from utils.io import dump_pickle
from dataclasses import dataclass

SplitData = NewType('SplitData', Dict[str, List[Dict[str, torch.Tensor]]])


# class Args(TypedArgs):

#     def __init__(self):
#         parser = argparse.ArgumentParser()

#         self.bboxes = parser.add_argument(
#             '--bboxes'
#         )
#         self.output = parser.add_argument(
#             '-o', '--output'
#         )
#         self.num_bboxes: int = parser.add_argument(
#             '-n', '--num-bboxes', type=int, default=10
#         )

#         self.parse_args_from(parser)


@dataclass
class Args(TypedArgs):
    bboxes: str = add_argument(
        '--bboxes',
        help='folder containing all split{n}.pt'
    )
    output: str = add_argument(
        '-o', '--output',
        help='output dir'
    )
    num_bboxes: int = add_argument(
        '-n', '--num-bboxes', default=5,
        help='10 for ablation study, 5 is enough'
    )


def load_bboxes(args: Args) -> List[SplitData]:

    splits = os.listdir(args.bboxes)

    # splits.remove('split5.pt.bak')

    splits = sorted(splits)
    print(splits)

    for split in tqdm(splits):
        data = torch.load(os.path.join(args.bboxes, split))
        yield data


def count_frames(args: Args):
    num_frames = 0
    for data in load_bboxes(args):
        for k, v in data.items():
            num_frames += len(v)
    print(f'total frames: {num_frames}')
    return num_frames


def main(args: Args):
    os.makedirs(args.output, exist_ok=True)

    num_frames = count_frames(args)

    data = load_bboxes(args)

    # scores = dict()
    # labels = dict()

    # fp_scores = open_memmap(
    #     os.path.join(args.output, 'scores.npy'),
    #     mode='w+',
    #     dtype=np.float32,
    #     shape=(num_frames, args.num_bboxes)
    # )
    # fp_labels = open_memmap(
    #     os.path.join(args.output, 'labels.npy'),
    #     mode='w+',
    #     dtype=np.int64,
    #     shape=(num_frames, args.num_bboxes)
    # )

    # We don't need scores and labels
    fp_bbox = open_memmap(
        os.path.join(args.output, 'data.npy'),
        mode='w+',
        dtype=np.float32,
        shape=(num_frames, args.num_bboxes, 4)
    )

    indices = dict()
    index = 0
    for split_data in data:

        for key, value in tqdm(split_data.items()):
            score_list = []
            label_list = []
            bbox_list = []

            num_frames = len(value)

            for frame in value:
                frame_labels = frame['labels']
                frame_scores = frame['scores']
                frame_bbox = frame['bbox']

                N = frame_labels.shape[0]
                N = min(N, args.num_bboxes)
                # print(frame_labels.shape)
                # print(frame_scores.shape)
                # exit()
                new_labels = torch.empty(
                    (args.num_bboxes,), dtype=frame_labels.dtype).fill_(-1)
                new_scores = torch.zeros((args.num_bboxes,))
                new_bbox = torch.zeros((args.num_bboxes, 4))

                new_labels[:N] = frame_labels[:N]
                new_scores[:N] = frame_scores[:N]
                new_bbox[:N] = frame_bbox[:N]

                label_list.append(new_labels)
                score_list.append(new_scores)
                bbox_list.append(new_bbox)

            labels = torch.stack(label_list).numpy()
            scores = torch.stack(score_list).numpy()
            bbox = torch.stack(bbox_list).numpy()

            indices[key] = [index, num_frames]
            # fp_scores[index: index + num_frames] = scores
            # fp_labels[index: index + num_frames] = labels
            fp_bbox[index: index + num_frames] = bbox

            index += num_frames

    # torch.save(scores, 'scores.pt')
    # torch.save(labels, 'labels.pt')

    del fp_bbox
    # del fp_labels
    # del fp_scores

    dump_pickle(indices, os.path.join(args.output, 'indices.pkl'))


if __name__ == "__main__":
    args = Args()
    main(args)
