import logging
import math
import os
import pickle
import random
from pprint import pprint
from typing import Dict

import numpy as np
import pandas as pd
# import ipdb
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory
from termcolor import colored
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from arguments import args
from dataset import get_dataloader
from model import get_model
from utils import MULTIPLE_CHOICE_TASKS, accuracy, count_correct
# from torchpie.config import config
# from torchpie.environment import args, experiment_path
# from torchpie.logging import logger
# from torchpie.meters import AverageMeter
# from torchpie.parallel import FakeObj
# from torchpie.utils.checkpoint import save_checkpoint
from utils.checkpoint import save_checkpoint
from utils.config import config
from utils.dictionary import CharDictionary, Dictionary
from utils.io import load_pickle
from utils.logging import set_default_logger
from utils.meters import AverageMeter

logger = logging.getLogger(__name__)


def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimzier: optim.Optimizer, epoch: int):
    loader_length = len(loader)

    losses = AverageMeter('Loss')

    if TASK in utils.MULTIPLE_CHOICE_TASKS or TASK in ['frameqa', 'youtube2text']:
        result = AverageMeter('Acc')
    else:
        result = AverageMeter('MSE')

    model.train()
    # for i, data in enumerate(loader):
    for i, data in enumerate(tqdm(loader)):

        data = utils.batch_to_gpu(data)

        (
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox,
            answer
        ) = data

        if config.get_bool('abc.is_multiple_choice'):
            answer = torch.zeros_like(answer)

        out = model(
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox
        )

        loss: torch.Tensor = criterion(out, answer)

        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        compute_score(losses, result, out, answer, loss)

        # logger.info(
        #     f'Train Epoch [{epoch}][{i}/{loader_length}]\t'
        #     f'{result}%\t{losses}'
        # )

        if args.debug:
            break

    writer.add_scalar(f'Train/{losses.name}', losses.avg, epoch)
    writer.add_scalar(f'Train/{result.name}', result.avg, epoch)


@torch.no_grad()
def test(model: nn.Module, loader: DataLoader, criterion: nn.Module, epoch: int) -> float:
    loader_length = len(loader)

    losses = AverageMeter('Loss')

    if TASK in utils.MULTIPLE_CHOICE_TASKS or TASK in ['frameqa', 'youtube2text']:
        result = AverageMeter('Acc')
    else:
        result = AverageMeter('MSE')

    type_meters = dict()
    if TASK == 'youtube2text':
        youtube2text_meters: Dict[int, AverageMeter] = dict()
        for qtype_id, qtype in youtube2text_qtype_dict.items():
            youtube2text_meters[qtype_id] = AverageMeter(qtype, fmt=':.3f')
        youtube2text_meters['other'] = AverageMeter('other', fmt=':.3f')

    model.eval()

    final_out = []

    for i, data in enumerate(tqdm(loader)):

        data = utils.batch_to_gpu(data)

        (
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox,
            answer
        ) = data

        if config.get_bool('abc.is_multiple_choice'):
            answer = torch.zeros_like(answer)

        out = model(
            question, question_length, question_chars,
            a1, a1_length, a1_chars,
            a2, a2_length, a2_chars,
            a3, a3_length, a3_chars,
            a4, a4_length, a4_chars,
            a5, a5_length, a5_chars,
            features, c3d_features, bbox_features, bbox
        )

        loss: torch.Tensor = criterion(out, answer)

        compute_score(losses, result, out, answer, loss)

        if TASK == 'youtube2text':
            corrects = out.argmax(dim=1).eq(answer)
            qtype_ids = question[:, 0]

            all_corrects = corrects.sum()
            all_questions = len(question)

            for qtype_id in youtube2text_qtype_dict.keys():
                qtype_meter = youtube2text_meters[qtype_id]

                current_qtype = qtype_ids.eq(qtype_id)
                num_questions = current_qtype.sum()

                if num_questions > 0:
                    currect_qtype_corrects = (
                        corrects & current_qtype).sum()

                    qtype_meter.update(
                        currect_qtype_corrects.float() / num_questions,
                        num_questions
                    )

                    all_corrects -= currect_qtype_corrects
                    all_questions -= num_questions

            if all_questions > 0:
                youtube2text_meters['other'].update(
                    all_corrects.float() / all_questions, all_questions)

        if args.debug:
            break

    writer.add_scalar(f'Test/{losses.name}', losses.avg, epoch)
    writer.add_scalar(f'Test/{result.name}', result.avg, epoch)

    if TASK == 'youtube2text':
        avg_per_class = 0

        for meter in youtube2text_meters.values():
            logger.info(f'Test Epoch [{epoch}] {meter}, n={meter.count}')
            avg_per_class += meter.avg

        avg_per_class /= 3

        logger.info(f'Test Epoch [{epoch}], Avg. Per-class: {avg_per_class}')

        for meter in youtube2text_meters.values():
            type_meters[meter.name] = meter.avg.item()

    return result.avg, type_meters


@torch.no_grad()
def compute_score(losses: AverageMeter, result: AverageMeter, out: torch.Tensor, answer: torch.Tensor,
                  loss: torch.Tensor):
    batch_size = answer.shape[0]

    if TASK in utils.MULTIPLE_CHOICE_TASKS or TASK in ['frameqa', 'youtube2text']:
        acc = accuracy(out, answer)
        result.update(acc.item(), batch_size)
    elif TASK == 'count':
        out = out * 10. + 1.
        mse = F.mse_loss(out.round().clamp(1., 10.), answer.clamp(1., 10.))
        result.update(mse.item(), batch_size)

    if TASK in MULTIPLE_CHOICE_TASKS or config.get_bool('abc.is_multiple_choice'):
        losses.update(loss.item() / batch_size, batch_size)
    else:
        losses.update(loss.item(), batch_size)


def main():
    best_result = math.inf if TASK == 'count' else 0.0
    best_type_meters = dict()

    train_loader, test_loader = get_dataloader(config, logger)

    num_classes = 1
    if TASK == 'frameqa':
        answer_dict = utils.load_answer_dict()
        num_classes = len(answer_dict)
    if TASK == 'youtube2text':
        if config.get_bool('abc.is_multiple_choice'):
            num_classes = 1
        else:
            num_classes = 1000
    logger.info(f'Num classes: {num_classes}')

    vocab_size = utils.get_vocab_size(config, TASK, level='word')
    char_vocab_size = utils.get_vocab_size(config, TASK, level='char')

    model = get_model(vocab_size, char_vocab_size, num_classes)
    model = model.cuda()

    if TASK in MULTIPLE_CHOICE_TASKS:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    elif TASK == 'count':
        inner_criterion = nn.MSELoss()

        def criterion(input, target):
            target = (target - 1.) / 10.
            return inner_criterion(input, target)

        # criterion = nn.SmoothL1Loss()
    elif TASK in ['frameqa']:
        criterion = nn.CrossEntropyLoss()

    elif TASK == 'youtube2text':
        if config.get_bool('abc.is_multiple_choice'):
            criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer_type = config.get_string('optimizer')

    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config.get_float('adam.lr'))
    else:
        raise Exception(f'Unknow optimizer: {optimizer_type}')

    start_epoch = 1
    end_epoch = config.get_int('num_epochs')

    for epoch in range(start_epoch, end_epoch + 1):
        logger.info(f'Epoch [{epoch}/{end_epoch}] start')

        train(model, train_loader, criterion, optimizer, epoch)
        current_result, current_type_meters = test(
            model, test_loader, criterion, epoch)

        logger.info(f'Epoch [{epoch}/{end_epoch}] end')

        if args.debug:
            break

        is_best = False
        if TASK == 'count':
            if current_result < best_result:
                is_best = True
                best_result = current_result

        else:
            if current_result > best_result:
                is_best = True
                best_result = current_result
                best_type_meters = current_type_meters

        logger.info(
            colored("Current best result: {:.2f}, Exp path: {}".format(best_result, args.experiment_path), "red"))
        logger.info(best_type_meters)
        save_checkpoint({
            'arch': config.get_string('arch'),
            'task': TASK,
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_result': best_result,
            'optimizer': optimizer.state_dict(),
            'best_type_meters': best_type_meters,
        }, is_best=is_best, folder=args.experiment_path)

    if TASK == 'count':
        logger.info(f'Best MSE: {best_result}')
    else:
        logger.info(f'Best Acc: {best_result}')


def fix_seed(config):
    seed = config.get_int('seed')
    logger.info(f'Set seed={seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":

    set_default_logger(args.experiment_path, debug=args.debug)
    # config = ConfigFactory.parse_file(args.config)

    fix_seed(config)

    pprint(config)

    TASK = config.get_string('task')

    best_meters = dict()

    if TASK == 'youtube2text':
        youtube2text_dictionary = Dictionary.load_from_file(
            os.path.join(
                config.get_string('cache_path'), 'youtube2text_dictionary.pkl'
            )
        )
        youtube2text_qtype_dict = dict()
        for qtype in ['what', 'who']:
            qtype_id = youtube2text_dictionary.word2idx[qtype]
            youtube2text_qtype_dict[qtype_id] = qtype

    if args.experiment_path is not None:
        writer = SummaryWriter(log_dir=args.experiment_path)
    else:
        # writer: SummaryWriter = FakeObj()
        raise Exception('No exp path for tensorboard')

    main()

    writer.close()
