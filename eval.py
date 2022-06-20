
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
import utils.misc as workspace
from torchvision.ops import sigmoid_focal_loss


def eval(data_loader, model, exp_dir):
    model.eval()
    test_image_savepath = os.path.join(workspace.get_dir(exp_dir, workspace.evaluation_subdir), "images")
    if not os.path.isdir(test_image_savepath):
        os.makedirs(test_image_savepath)
    pbar = tqdm(data_loader)
    for _, data in enumerate(pbar):
        img = data['image'].cuda()
        # lab = data['mask'].cuda()
        name = data['name']
        outimg_f = os.path.join(test_image_savepath, f"testout_{name}.png")
        with torch.no_grad():
            inp_logits = model(img)
            output = torch.sigmoid(inp_logits).data.cpu().numpy().squeeze()
        imageio.imsave(outimg_f, img_as_ubyte(output))


def main(args):
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)

    if not os.path.isdir(args.exp_dir):
        print(f"[ERROR] Experiment dir {args.exp_dir} does not exist!")
        sys.exit(0)
    experiment_dir = args.exp_dir
    checkpoint = args.checkpoint

    with open(os.path.join(experiment_dir, "specs.json"), "r") as f:
        specs = json.load(f)
    train_data_dir = specs["DataSource"]
    learning_rate = specs["LearningRate"]
    num_epochs = specs["Epochs"]
    batch_size = specs["BatchSize"]
    if_debug = specs["Debug"]
    n_debug = specs["NumDebug"]

    print(
        f'Learning Rate:{learning_rate} | Epochs:{num_epochs} | BatchSize:{batch_size}')
    print(f"Training data dir: {train_data_dir}")

    # channels = (1, 64, 128, 256, 512, 1024)
    # model = Unet2D(**specs["NetworkSpecs"])
    channels = 1
    classes = 1
    model = Unet2D(channels, num_class=classes)
    model = model.cuda()
    workspace.load_model_checkpoint(experiment_dir, checkpoint, model)

    trn_img_dir = os.path.join(train_data_dir, "images")
    trn_lab_dir = os.path.join(train_data_dir, "labels")
    eval_loader = sk_loader(trn_img_dir, trn_lab_dir,
                            batch_size=1, debug=if_debug, num_debug=n_debug)

    print("Evaluating model on training set....")
    eval(eval_loader, model, experiment_dir)


if __name__ == "__main__":
    # init parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', '-e', type=str,
                        default="./experiments/init_run/", help="Path to hyperparams dir.")
    parser.add_argument('--checkpoint', '-c', type=str,
                        default="latest", help="Which checkpoint to eval on?")
    args = parser.parse_args()
    main(args)
