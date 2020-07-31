import argparse
import os
import pickle
from glob import glob
from pprint import pprint

import numpy as np
import torch
import torchvision.transforms as T
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import ResNet, resnet152
from torchvision.models.vgg import VGG, vgg16_bn
from tqdm import tqdm
# from typed_args import TypedArgs
import json
from torchvision.ops import roi_align
from collections import defaultdict
from typing import List, Tuple, Dict
from torch import nn
from io import BytesIO
from utils.io import load_pickle


# class Args(TypedArgs):

#     def __init__(self):
#         parser = argparse.ArgumentParser()

#         self.input_dir = parser.add_argument('-i', '--input-dir')
#         self.output_dir = parser.add_argument(
#             '-o', '--output-dir', default='data/resnet152_layer4_features')

#         self.batch_size = parser.add_argument(
#             '-b', '--batch-size', type=int, default=512
#         )

#         self.num_workers = parser.add_argument(
#             '-n', '--num-workers', type=int, default=4
#         )
#         self.part = parser.add_argument(
#             '-p', '--part'
#         )
#         self.num_boxes = parser.add_argument(
#             '--num-boxes', type=int
#         )
#         self.arch = parser.add_argument(
#             '-a', '--arch', help='vgg16, c3d = 4096, resnet = 2048', default='resnet152'
#         )
#         self.frame_path = parser.add_argument(
#             '-f', '--frame-path', help='MSVD'
#         )

#         self.parse_args_from(parser)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir')
parser.add_argument(
    '-o', '--output-dir', default='data/resnet152_layer4_features'
)

parser.add_argument(
    '-b', '--batch-size', type=int, default=512
)

parser.add_argument(
    '-n', '--num-workers', type=int, default=4
)
parser.add_argument(
    '-p', '--part'
)
parser.add_argument(
    '--num-boxes', type=int
)
parser.add_argument(
    '-a', '--arch', help='vgg16, c3d = 4096, resnet = 2048', default='resnet152'
)
parser.add_argument(
    '-f', '--frame-path', help='MSVD only'
)
parser.add_argument(
    '--msvd', action='store_true', help='MSVD uses different frame naming strategy'
)


class VideoDataset(Dataset):

    def __init__(self, args, keys: Dict[str, int], extension='*.jpg'):
        self.args = args
        self.root = args.input_dir
        self.part = args.part
        self.extension = extension

        # with open(self.part, 'rb') as f:
        #     self.videos = torch.load(BytesIO(f.read()))

        self.video_dict = defaultdict(list)

        if args.arch == 'resnet152':
            for gif_name, num_frames in keys.items():
                for i in range(num_frames):
                    self.video_dict[gif_name].append(
                        os.path.join(self.root, gif_name, f'{i}.jpg')
                    )
        elif args.arch == 'vgg16':
            samples: List[str] = load_pickle(args.frame_path)
            videos = []
            for sample in tqdm(samples):
                gif_name = sample.split('/')[-1]
                videos.append(gif_name)
                num_frames = len(os.listdir(sample))

                selected_frames = np.linspace(
                    0, num_frames, 20 + 2)[1:20 + 1].astype(np.int) + 1
                for n in selected_frames:
                    if args.msvd:
                        frame_path = os.path.join(
                            sample, f'{n + 1:06}.jpg')  # For MSVD-QA
                    else:
                        frame_path = os.path.join(
                            sample, f'{n}.jpg')  # For TGIF-QA

                    self.video_dict[gif_name].append(
                        # os.path.join(sample, f'{n}.jpg') # For TGIF-QA
                        # os.path.join(sample, f'{n + 1:06}.jpg')  # For MSVD-QA
                        frame_path
                    )

        self.samples = list()
        self.indices = dict()

        index = 0
        for key, value in self.video_dict.items():
            self.samples.extend(value)
            num_frames = len(value)
            self.indices[key] = [index, num_frames]
            index += num_frames

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print('self.video_dict')
        pprint(self.video_dict[gif_name])

        # del self.videos

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = Image.open(sample).convert('RGB')
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.samples)

    def save_indices(self, path: str):
        with open(os.path.join(path, 'indices.pkl'), 'wb') as f:
            pickle.dump(self.indices, f)


# def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> (torch.Tensor, List[torch.Tensor]):
#     images, boxes = zip(*batch)

#     images = torch.stack(images)

#     return images, boxes


def get_model(args):
    if args.arch == 'resnet152':
        def my_forward(self: ResNet, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            # x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            # x = self.fc(x)

            return x

        model = resnet152(pretrained=True).cuda()
        model.forward = my_forward.__get__(model, ResNet)
        model.eval()
    elif args.arch == 'vgg16':

        full_model = vgg16_bn(pretrained=True).cuda()
        full_model.eval()
        model = full_model.features

    else:
        raise Exception

    return model


def get_bbox(args):
    bboxes: Dict[str, List[Dict[str, torch.Tensor]]] = torch.load(args.part)

    keys = dict()

    box_list = list()

    for gif_name, frames in tqdm(bboxes.items()):

        keys[gif_name] = len(frames)

        for frame in frames:
            bbox = frame['bbox']

            new_boxes = torch.zeros((args.num_boxes, 4))
            N, _ = bbox.shape
            N = min(args.num_boxes, N)

            # Resize to 7x7
            new_boxes[:N] = bbox[:N] * 7.

            box_list.append(new_boxes)

    return box_list, keys


# class RoiModel(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.resnet = get_model()

#     def forward(self, images: torch.Tensor, boxes: List[torch.Tensor]):
#         output = self.resnet(images)
#         output = roi_align(output, boxes, (1, 1))
#         output = output
#         return output


def get_dataloader(args, keys: Dict[str, int]):
    dataset = VideoDataset(args, keys)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        # collate_fn=collate_fn
    )

    return loader


def extract_features(args):
    model = get_model(args)
    # model = RoiModel()
    # model = nn.DataParallel(model)
    # model = model.cuda()
    # model.eval()

    bboxes, keys = get_bbox(args)

    loader = get_dataloader(args, keys)

    N = len(loader.dataset)

    # out_channels = 2048 if args.arch == 'resnet152' else 512
    if args.arch == 'resnet152':
        out_channels = 2048
    elif args.arch == 'vgg16':
        out_channels = 512

    fp = open_memmap(
        os.path.join(args.output_dir, 'data.npy'),
        mode='w+',
        dtype=np.float32,
        shape=(N, args.num_boxes, out_channels)
    )

    with torch.no_grad():
        for i, images in tqdm(enumerate(loader), total=len(loader)):
            images = images.cuda()

            output = model(images)

            current_batch_size = images.shape[0]

            current_index = i * args.batch_size
            current_boxes = bboxes[current_index: current_index +
                                   current_batch_size]
            current_boxes = [b.cuda() for b in current_boxes]
            output = roi_align(output, current_boxes, (1, 1))

            # index = i * args.batch_size
            # import ipdb; ipdb.set_trace()
            fp[current_index: current_index + current_batch_size] = output.view(
                current_batch_size, args.num_boxes, out_channels).cpu().numpy()

    print(fp[N - 1])
    del fp

    loader.dataset.save_indices(args.output_dir)


if __name__ == "__main__":
    # args = Args()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    extract_features(args)
