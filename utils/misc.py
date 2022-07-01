import os
import torch
import numpy as np


checkpoint_subdir = "checkpoints"
evaluation_subdir = "evals"
optim_param_subdir = "optim_parameters"


def get_dir(exp_dir, subdir):
    dir = os.path.join(exp_dir, subdir)
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return dir


def save_latest(exp_dir, epoch, model, optimizer):
    save_model(exp_dir, "model_checkpoint_latest.pth", model, epoch)
    save_optimizer(exp_dir, "optimizer_latest.pth", optimizer, epoch)


def save_checkpoint(exp_dir, epoch, model, optimizer):
    save_model(exp_dir, f"model_checkpoint_{epoch}.pth", model, epoch)
    save_optimizer(exp_dir, f"optimizer_{epoch}.pth", optimizer, epoch)


def save_model(exp_dir, fname, model, epoch):
    model_dir = get_dir(exp_dir, checkpoint_subdir)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_dir, fname)
    )


def load_model_checkpoint(exp_dir, checkpoint, model):
    state_f = os.path.join(get_dir(exp_dir, checkpoint_subdir),
                           f"model_checkpoint_{checkpoint}.pth")
    if not os.path.isfile(state_f):
        raise Exception(f'Model state dict "{state_f}" does not exist')
    data = torch.load(state_f)
    model.load_state_dict(data["model_state_dict"])
    return data["epoch"]


def save_optimizer(exp_dir, fname, optimizer, epoch):
    optim_dir = get_dir(exp_dir, optim_param_subdir)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optim_dir, fname)
    )


def load_optimizer(exp_dir, checkpoint, optimizer):
    state_f = os.path.join(get_dir(exp_dir, optim_param_subdir),
                            f"optimizer_{checkpoint}.pth")
    if not os.path.isfile(state_f):
        raise Exception(f'optimizer state dict "{state_f}" does not exist')
    data = torch.load(state_f)
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data["epoch"]
