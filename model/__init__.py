from torch import nn
from pyhocon import ConfigTree
from .lgcn import LGCN
# from torchpie.config import config
# from torchpie.logging import logger
from utils.config import config
import logging

logger = logging.getLogger(__name__)


def get_model(
        vocab_size: int,
        char_vocab_size: int,
        num_classes: int
) -> nn.Module:
    arch = config.get_string('arch')
    logger.info(f'Using model: {arch}')

    if arch == 'abc':
        opt = config.get_config('abc')
        opt.put('vocab_size', vocab_size)
        opt.put('num_classes', num_classes)
        opt.put('char_vocab_size', char_vocab_size)
        model = LGCN(opt)

    else:
        raise Exception(f'No such arch: {arch}')

    return model
