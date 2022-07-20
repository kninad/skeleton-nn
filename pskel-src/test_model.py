import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# from models.skel_point_net import SkelPointNet
from SkelPointNet import SkelPointNet
from DataUtil import PCDataset
import FileRW as rw

sys.path.append("..")
import utils.misc as workspace


def parse_arguments():
    parser = argparse.ArgumentParser(description="Point2Skeleton")

    parser.add_argument(
        "--exp_dir",
        "-e",
        type=str,
        default="../experiments/init_run/",
        help="Path to hyperparams dir.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="val_split.txt",
        help="Dataset split file for validation. Should be present in the specified experiment dir.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="latest",
        help="Which model checkpoint to evaluate on?",
    )
    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")
    args = parser.parse_args()
    return args


def save_results(log_path, batch_id, input_xyz, skel_xyz, skel_r):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    for i in range(batch_size):
        save_name_input = os.path.join(log_path, f"val_{batch_id[i]}_input.ply")
        save_name_sphere = os.path.join(log_path, f"val_{batch_id[i]}_sphere.obj")
        save_name_center = os.path.join(log_path, f"val_{batch_id[i]}_center.ply")
        rw.save_ply_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_ply_points(skel_xyz_save[i], save_name_center)


def main(args):
    experiment_dir = args.exp_dir
    split_file = args.split
    checkpoint = args.checkpoint
    gpu = args.gpu

    # Setup the checkpoint and model eval dirs in exp_dir
    checkpt_dir = os.path.join(experiment_dir, workspace.checkpoint_subdir)
    eval_dir = os.path.join(experiment_dir, workspace.evaluation_subdir, "testing")
    rw.check_and_create_dirs([checkpt_dir, eval_dir])
    with open(os.path.join(experiment_dir, "specs.json"), "r") as f:
        specs = json.load(f)

    data_dir = specs["DataSource"]
    if not os.path.isdir(data_dir):
        print(
            f"The provided data dir in specs.json is invalid! {data_dir} is not found on this system."
        )
        print("Exiting...")
        return 0

    to_normalize = specs["Normalize"]
    point_num = specs["InputPointNum"]
    skelpoint_num = specs["SkelPointNum"]
    # Assume Training/Test split file (given as cmd line arg) will be present in the experiment dir
    pc_list_file = os.path.join(experiment_dir, split_file)

    # load networks
    model_skel = SkelPointNet(
        num_skel_points=skelpoint_num, input_channels=0, use_xyz=True
    )

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print("GPU Number:", torch.cuda.device_count(), "GPUs!")
        model_skel.cuda()
        model_skel.eval()
    else:
        print("No CUDA detected.")
        sys.exit(0)

    # Load the saved model
    model_epoch = workspace.load_model_checkpoint(
        experiment_dir, checkpoint, model_skel
    )
    print(f"Evaluating model on using checkpoint={checkpoint} and epoch={model_epoch}.")

    # load data and evaluate
    pc_list = rw.load_data_id(pc_list_file)
    train_data = PCDataset(pc_list, data_dir, point_num, to_normalize)
    data_loader = DataLoader(
        dataset=train_data, batch_size=1, shuffle=False, drop_last=False
    )

    for _, batch_data in enumerate(tqdm(data_loader)):
        batch_id, batch_pc = batch_data
        batch_id = batch_id
        batch_pc = batch_pc.cuda().float()
        with torch.no_grad():
            skel_xyz, skel_r, _ = model_skel(batch_pc, compute_graph=False)
        save_results(eval_dir, batch_id, batch_pc, skel_xyz, skel_r)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
