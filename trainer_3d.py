import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import img_as_ubyte
from matplotlib import pyplot as plt

from monai.utils import set_determinism
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import first
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceFocalLoss, FocalLoss
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.transforms import AsDiscrete

from utils.data import get_surf_srep_split, get_srep_data_transform, get_aug_transform
import utils.misc as workspace


def train(train_loader, model, optimizer, loss_fn1, loss_fn2,  epochs,
          exp_dir, save_every, device, start_epoch=1):
    print("Begin Training.......")
    writer = SummaryWriter(os.path.join(exp_dir, "tb_logs"))

    epoch_loss_values = list()

    for epoch in tqdm(range(start_epoch, epochs+1)):
        print("-" * 10)
        print(f"epoch {epoch}/{epochs}")

        # Training step
        model.train()
        epoch_loss = 0
        epoch_loss_dice = 0
        epoch_loss_focal = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_dice = loss_fn1(outputs, labels)
            loss_focal = loss_fn2(outputs, labels)
            loss = loss_dice + loss_focal
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_dice += loss_dice.item()
            epoch_loss_focal += loss_focal.item()

        epoch_loss /= step
        epoch_loss_dice /= step
        epoch_loss_focal /= step
        epoch_loss_values.append(epoch_loss)
        writer.add_scalar("Loss/overall", epoch_loss, epoch)
        writer.add_scalar("Loss/dice", epoch_loss_dice, epoch)
        writer.add_scalar("Loss/focal", epoch_loss_focal, epoch)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        workspace.save_latest(exp_dir, epoch, model, optimizer)
        if epoch % save_every == 0:
            workspace.save_checkpoint(exp_dir, epoch, model, optimizer)

    writer.flush()


def main(args):

    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    set_determinism(seed=2020)
    if not os.path.isdir(args.exp_dir):
        print(f"[ERROR] Experiment dir {args.exp_dir} does not exist!")
        return 0
    experiment_dir = args.exp_dir
    continue_from = args.continue_from

    # Setup the checkpoint and model eval dirs in exp_dir
    checkpt_dir = os.path.join(experiment_dir, workspace.checkpoint_subdir)
    eval_dir = os.path.join(experiment_dir, workspace.evaluation_subdir)
    if not os.path.isdir(checkpt_dir):
        os.makedirs(checkpt_dir)
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    with open(os.path.join(experiment_dir, "specs.json"), "r") as f:
        specs = json.load(f)
    train_data_dir = specs["DataSource"]
    if not os.path.isdir(train_data_dir):
        print(
            f"The provided data dir in specs.json is invalid! {train_data_dir} is not found on this system.")
        return 0

    learning_rate = specs["LearningRate"]
    num_epochs = specs["Epochs"]
    save_epoch = specs["SaveEvery"]
    batch_size = specs["BatchSize"]
    if_debug = specs["Debug"]
    resize_shape = specs["ResizeShape"]
    num_data_loaders = specs["NumDataLoaders"]
    print(
        f'Learning Rate:{learning_rate} | Epochs:{num_epochs} | BatchSize:{batch_size}')
    print(f"Training data dir: {train_data_dir}")

    # data_transforms = get_srep_data_transform((resize_shape, resize_shape, resize_shape))
    data_transforms = get_aug_transform(resize_shape=(resize_shape, resize_shape, resize_shape))
    
    trn_files, _, _ = get_surf_srep_split(
        train_data_dir, random_shuffle=False, debug=if_debug)

    trn_ds = CacheDataset(data=trn_files, transform=data_transforms,
                          cache_rate=0.5, num_workers=num_data_loaders)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_data_loaders)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model).cuda()

    criterion1 = DiceLoss(sigmoid=True)
    criterion2 = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    if continue_from is not None:
        model_epoch = workspace.load_model_checkpoint(
            experiment_dir, continue_from, model)
        optim_epoch = workspace.load_optimizer(
            experiment_dir, continue_from, optimizer)
        if model_epoch != optim_epoch:
            raise RuntimeError(
                f"Checkpoint Epoch mismatch: model={model_epoch} vs optimizer={optim_epoch}")
        starting_epoch = model_epoch + 1
        print(f"Resuming training from saved checkpoint:") 
        print(f"{continue_from} checkpoint | {starting_epoch} epoch.\n")        
    else:
        starting_epoch = 1

    train(trn_loader, model, optimizer,
          criterion1, criterion2, num_epochs, experiment_dir, save_epoch, device, starting_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', '-e', type=str,
                        default="./experiments/init_run/", help="Path to hyperparams dir.")
    parser.add_argument('--continue', '-c', type=str,
                        dest="continue_from",
                        help="Model checkpoint to continue the"
                        + "training process from. Can be 'latest' or an int (for epoch number)")
    args = parser.parse_args()
    main(args)
