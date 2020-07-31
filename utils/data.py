import os
import pickle
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from pprint import pprint
from tqdm import tqdm


class VideoDataset(Dataset):

    def __init__(self, root: str, extension='*.jpg'):
        self.root = root
        self.extension = extension

        self.videos = os.listdir(root)

        def by_index(path: str):
            basename = os.path.basename(path)
            index = os.path.splitext(basename)[0]
            return int(index)

        self.video_dict = dict()
        for video in tqdm(self.videos):
            self.video_dict[video] = sorted(
                glob(os.path.join(root, video, extension)),
                key=by_index
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
            # T.Resize((300, 300), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print('self.video_dict')
        pprint(self.video_dict[video])

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
