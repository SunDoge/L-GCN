import json
import pickle
from PIL import Image, features
# from torchpie.logging import logger
import logging

logger = logging.getLogger(__name__)

logger.info(f'pillow version: {Image.PILLOW_VERSION}')
logger.info(
    'Using jpeg-turbo: {}'.format(features.check_feature('libjpeg_turbo')))


def load_json(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(obj, filename: str):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')
