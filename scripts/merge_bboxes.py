import torch
# from typed_args import TypedArgs
import argparse
import os
from typing import Dict, List, NewType, Tuple
from tqdm import tqdm
from numpy.lib.format import open_memmap
import numpy as np
from utils.npy_file import NpyFile
from utils.io import dump_pickle


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

parser = argparse.ArgumentParser()
parser.add_argument('--bboxes', help='path to bboxes')
parser.add_argument('-o', '--output', help='output path')
parser.add_argument('-n', '--num-bboxes', type=int, default=5,
                    help='use N bboxes, 5 is enough, 10 for ablation study')


def load_bboxes(args: Args) -> List[NpyFile]:

    splits = os.listdir(args.bboxes)
    splits = sorted(splits)
    print(splits)

    fps = []
    for split in tqdm(splits):
        fp = NpyFile(os.path.join(args.bboxes, split))
        fps.append(fp)

    return fps


def get_new_indices(fp: NpyFile, index: int) -> Dict[str, Tuple[int, int]]:
    indices = fp.indices

    for k in indices.keys():
        indices[k][0] += index

    return indices


def count_frames(fps: List[NpyFile]) -> int:
    res = 0

    for fp in fps:
        res += len(fp.data)

    return res


def main(args):
    os.makedirs(args.output, exist_ok=True)

    fps = load_bboxes(args)

    total_frames = count_frames(fps)

    print('total_frames:', total_frames)

    new_indices = dict()
    new_fp = open_memmap(
        os.path.join(args.output, 'data.npy'),
        mode='w+',
        dtype=np.float32,
        shape=(total_frames, 10, 2048)
    )

    index = 0
    for fp in tqdm(fps):
        length = len(fp.data)
        new_fp[index: index + length] = fp.data

        new_indices.update(get_new_indices(fp, index))

        index += length

    del new_fp
    dump_pickle(new_indices, os.path.join(args.output, 'indices.pkl'))


if __name__ == "__main__":
    # args = Args()
    args = parser.parse_args()
    main(args)
