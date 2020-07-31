import os
import torch
import shutil

def save_checkpoint(state: dict, is_best=False, folder='', filename='checkpoint.pth.tar'):
    filename = os.path.join(folder, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(folder, 'model_best.pth.tar'))
