import argparse
import os
from collections import defaultdict
from typing import Dict, List

import ipdb
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from yacs.config import CfgNode

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from utils.data import VideoDataset
from utils.io import dump_pickle, load_image, load_pickle
import numpy as np


# MSVD = True


# class Args(TypedArgs):

#     def __init__(self):

#         parser = argparse.ArgumentParser()
#         self.frame_path = parser.add_argument(
#             '-f', '--frame-path',
#         )
#         self.output = parser.add_argument(
#             '-o', '--output'
#         )
#         self.batch_size = parser.add_argument(
#             '-b', '--batch-size', type=int, default=16
#         )
#         self.num_workers = parser.add_argument(
#             '-n', '--num-workers', type=int, default=4
#         )

#         self.parse_args_from(parser)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--frame-path', help='path to frames')
parser.add_argument('-o', '--output', help='path to output pickle file')
parser.add_argument('-b', '--batch-size', default=16)
parser.add_argument('-n', '--num-workers', type=int, default=4)
parser.add_argument('--msvd', action='store_true', help='for MSVD-QA')
parser.add_argument('-c', '--config', help='path to e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml')


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class FrameDataset(Dataset):

    def __init__(self, cfg: CfgNode, samples: List[str]):
        self.cfg = cfg
        self.transform = self.build_transform()
        self.samples: List[str] = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = load_image(sample)
        image = self.transform(image)
        return image

    def build_transform(self):
        cfg = self.cfg

        to_bgr = T.Lambda(lambda x: x[[2, 1, 0]] * 255)

        normalize = T.Normalize(
            cfg.INPUT.PIXEL_MEAN,
            cfg.INPUT.PIXEL_STD
        )
        transform = T.Compose([
            Resize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
            T.ToTensor(),
            to_bgr,
            normalize
        ])

        return transform

    # def set_samples(self, samples):
    #     self.samples = samples


class GIFDataset(Dataset):

    def __init__(self, cfg: CfgNode, args):
        self.cfg = cfg
        self.args = args
        self.samples: List[str] = load_pickle(args.frame_path)

        self.video_dict: Dict[str, List[str]] = defaultdict(list)
        self.videos = []

        for sample in tqdm(self.samples):
            gif_name = sample.split('/')[-1]
            self.videos.append(gif_name)
            num_frames = len(os.listdir(sample))
            # if MSVD:
            if args.msvd:
                selected_frames = np.linspace(
                    0, num_frames, 20 + 2)[1:20+1].astype(np.int) + 1
                for n in selected_frames:
                    self.video_dict[gif_name].append(
                        # os.path.join(sample, f'{n}.jpg') # For TGIF-QA
                        os.path.join(sample, f'{n + 1:06}.jpg')  # For MSVD-QA
                    )

            else:
                for n in range(num_frames):
                    self.video_dict[gif_name].append(
                        os.path.join(sample, f'{n}.jpg') # For TGIF-QA
                        # os.path.join(sample, f'{n + 1:06}.jpg')  # For MSVD-QA
                    )

        # self.frame_dataset = FrameDataset(cfg)

    def __getitem__(self, index):
        gif_name = self.videos[index]
        # self.frame_dataset.set_samples(self.video_dict[gif_name])
        dataset = FrameDataset(self.cfg, self.video_dict[gif_name])
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        return loader, gif_name

    def __len__(self):
        return len(self.samples)


def collate_fn(batch: List[torch.Tensor]) -> ImageList:
    return to_image_list(batch, size_divisible=cfg.DATALOADER.SIZE_DIVISIBILITY)


class Extractor:

    def __init__(self, cfg: CfgNode, args):
        self.args = args
        self.cfg = cfg.clone()
        self.model: nn.Module = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        # Load weight
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(
            cfg, self.model, save_dir=save_dir
        )
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.cpu_device = torch.device('cpu')
        # Keep all result
        self.confidence_threshold = 0.

        self.datasets = GIFDataset(cfg, args)

        self.results = defaultdict(list)

    @torch.no_grad()
    def compute_predictions(self, image_list: ImageList) -> List[BoxList]:
        image_list = image_list.to(self.device)
        predictions = self.model(image_list)
        return predictions

    def run_once(self):

        for dataset, gif_name in self.datasets:
            # ipdb.set_trace()
            loader = DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn
            )

            for image_list in loader:
                predictions = self.compute_predictions(image_list)

                self.save_predictions(gif_name, predictions)

                ipdb.set_trace()

                break

            break

        self.dump_result()

    def run(self):
        dataset_loader = DataLoader(
            self.datasets,
            batch_size=1,
            collate_fn=lambda x: x[0]
        )
        for loader, gif_name in tqdm(dataset_loader):
            # ipdb.set_trace()

            for image_list in tqdm(loader):
                predictions = self.compute_predictions(image_list)

                self.save_predictions(gif_name, predictions)

        self.dump_result()

    def save_predictions(self, gif_name: str, predictions: List[BoxList]):
        # preds = [
        #     {
        #         'bbox': pred.bbox,
        #         'scores': pred.scores,
        #         'labels': pred.labels,
        #     }
        #     for pred in predictions
        # ]

        for pred in predictions:
            pred: BoxList = pred.to(self.cpu_device).resize((1, 1))
            self.results[gif_name].append(
                {
                    'bbox': pred.bbox,
                    'scores': pred.get_field('scores'),
                    'labels': pred.get_field('labels'),
                }
            )

    def dump_result(self):
        torch.save(self.results, self.args.output)


def load_config(config_file='/home/huangdeng/Code/python/maskrcnn/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml') -> CfgNode:
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.MASK_ON", False])
    return cfg


def main(args):
    cfg = load_config(args.config)
    extractor = Extractor(cfg, args)
    # extractor.run_once()
    extractor.run()


if __name__ == "__main__":
    # args = Args()
    args = parser.parse_args()
    main(args)
