from logging import Logger

from pyhocon import ConfigTree
from torch.utils.data import DataLoader
# from torchpie.environment import args
from arguments import args

from .tgifqa_dataset import collate_fn


def get_dataloader(opt: ConfigTree, logger: Logger) -> (DataLoader, DataLoader):
    if opt.get_string('task') in ['msvd', 'youtube2text']:
        from .msvd_dataset import get_dataset
    else:
        from .tgifqa_dataset import get_dataset

    train_set, test_set = get_dataset(opt.get_config('dataset'), logger)

    train_loader = DataLoader(
        train_set,
        batch_size=opt.get_int('batch_size'),
        shuffle=True,
        num_workers=0 if args.debug else opt.get_int('num_workers'),
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set,
        batch_size=opt.get_int('batch_size'),
        shuffle=False,
        num_workers=0 if args.debug else opt.get_int('num_workers'),
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader
