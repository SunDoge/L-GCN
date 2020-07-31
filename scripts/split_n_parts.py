import os
import numpy as np
from typing import List
from utils.io import dump_pickle
import argparse
from pathlib import Path


# FRAME_PATH = '/mnt/dataset/tgif-qa/frames/'

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f', '--frame-path', default='data/tgif/frames',
    help='path to frames'
)
parser.add_argument(
    '-o', '--output', default='data/tgif/frame-splits/', type=Path,
    help='path to save the splited pickle file'
)
parser.add_argument(
    '-n', '--num-parts', default=6, type=int,
    help='split into N parts'
)


def get_all_gifs(frame_path: str):
    gifs = os.listdir(frame_path)
    gif_paths = [os.path.join(frame_path, gif) for gif in gifs]
    return gif_paths


def split_n_parts(gifs: List[str], n: int = 4):

    parts = np.array_split(gifs, n)

    return parts


if __name__ == "__main__":
    args = parser.parse_args()
    gifs = get_all_gifs(args.frame_path)
    # Split into N parts
    parts = split_n_parts(gifs, args.num_parts)
    for i, part in enumerate(parts):
        dump_pickle(list(part), args.output / f'split{i}.pkl')
