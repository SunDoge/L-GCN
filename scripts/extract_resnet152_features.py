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
from torchvision.models.resnet import ResNet, resnet152, resnet101
from torchvision.models.vgg import VGG, vgg16_bn
from tqdm import tqdm
# from typed_args import TypedArgs
from utils.data import VideoDataset
# from maskrcnn_benchmark.structures.bounding_box import BoxList


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

#         self.parse_args_from(parser)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir', help='path to tgif frames')
parser.add_argument('-o', '--output-dir', default='data/tgif/resnet152_pool5_features',
                    help='path to save the output features')
parser.add_argument('-b', '--batch-size', type=int, default=512)
parser.add_argument('-n', '--num-workers', type=int, default=4)


class MyDataset:

    def __init__(self):
        pass


def get_model():
    def my_forward(self: ResNet, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    # model = resnet152(pretrained=True).cuda()
    model = resnet101(pretrained=True).cuda()
    model.forward = my_forward.__get__(model, ResNet)
    model.eval()

    return model


def get_dataloader(args):
    dataset = VideoDataset(args.input_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return loader


def extract_features(args):

    model = get_model()
    loader = get_dataloader(args)

    N = len(loader.dataset)

    fp = open_memmap(
        os.path.join(args.output_dir, 'data.npy'),
        mode='w+',
        dtype=np.float32,
        shape=(N, 2048)
    )

    with torch.no_grad():
        for i, images in tqdm(enumerate(loader), total=len(loader)):
            images = images.cuda()
            output = model(images)

            current_batch_size = images.shape[0]

            index = i * args.batch_size

            fp[index: index + current_batch_size] = output.cpu().numpy()

    print(fp[N-1])
    del fp

    loader.dataset.save_indices(args.output_dir)


if __name__ == "__main__":
    # args = Args()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    extract_features(args)
