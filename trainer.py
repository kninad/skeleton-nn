import os
import time
import pickle
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import img_as_ubyte
import imageio

from models import Unet2D_simple as Unet2D
from utils.data import sk_loader
from utils.loss import WeightedFocalLoss, DiceLoss
from torchvision.ops import sigmoid_focal_loss


def train(train_loader, model, optimizer, loss_fn, epochs, checkpoint_dir, save_every, writer):
    for epoch in tqdm(range(1, epochs+1)):
        model.train(True)
        running_loss = 0.
        dice_loss = 0.
        focal_loss = 0.
        pbar = tqdm(train_loader)
        for _, data in enumerate(pbar):
            optimizer.zero_grad()
            img = data['image'].cuda()
            mask = data['mask'].cuda()
            inp_logits = model(img)
            loss1 = loss_fn(inp_logits, mask)
            loss2 = sigmoid_focal_loss(inp_logits, mask, reduction='mean')
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            dice_loss += loss1.item()
            focal_loss += loss2.item()

        writer.add_scalar("Loss/overall", running_loss, epoch)
        writer.add_scalar("Loss/dice", dice_loss, epoch)
        writer.add_scalar("Loss/focal", focal_loss, epoch)

        print(f"Epoch:{epoch} | Running Loss:{running_loss}")
        latest_model = f"model_checkpoint_latest.pth"
        torch.save(model.state_dict(), os.path.join(
            checkpoint_dir, latest_model))
        if epoch % save_every == 0:
            saved_model = f"model_checkpoint_{epoch}.pth"
            print(f"Saving model: {saved_model}")
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, saved_model))
    writer.flush()


def main(args):

    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)

    if not os.path.isdir(args.exp_dir):
        print(f"[ERROR] Experiment dir {args.exp_dir} does not exist!")
        sys.exit(0)
    experiment_dir = args.exp_dir

    # Setup the checkpoint and model eval dirs in exp_dir
    checkpt_dir = os.path.join(experiment_dir, "checkpoints")
    eval_dir = os.path.join(experiment_dir, "evals")
    if not os.path.isdir(checkpt_dir):
        os.makedirs(checkpt_dir)
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    with open(os.path.join(experiment_dir, "specs.json"), "r") as f:
        specs = json.load(f)
    train_data_dir = specs["DataSource"]
    learning_rate = specs["LearningRate"]
    num_epochs = specs["Epochs"]
    model_save = specs["SaveEvery"]
    batch_size = specs["BatchSize"]
    if_debug = specs["Debug"]
    n_debug = specs["NumDebug"]

    print(
        f'Learning Rate:{learning_rate} | Epochs:{num_epochs} | BatchSize:{batch_size}')
    print(f"Training data dir: {train_data_dir}")

    # channels = (1, 64, 128, 256, 512, 1024)
    channels = 1
    classes = 1
    model = Unet2D(channels, num_class=classes)
    model = model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, learning_rate)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = WeightedFocalLoss(alpha=[50, 0.75])
    criterion = DiceLoss()

    trn_img_dir = os.path.join(train_data_dir, "images")
    trn_lab_dir = os.path.join(train_data_dir, "labels")
    train_loader = sk_loader(trn_img_dir, trn_lab_dir,
                             batch_size=batch_size, debug=if_debug, num_debug=n_debug)
    print("Begin Training.......")
    writer = SummaryWriter(experiment_dir)
    train(train_loader, model, optimizer,
          criterion, num_epochs, checkpt_dir, model_save, writer)


if __name__ == "__main__":
    # init parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', '-e', type=str,
                        default="./experiments/init_run/", help="Path to hyperparams dir.")
    args = parser.parse_args()
    main(args)
