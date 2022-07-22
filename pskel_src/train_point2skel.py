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
        default="train_split.txt",
        help="Dataset split file for training. Should be present in the specified experiment dir.",
    )

    parser.add_argument(
        "--continue",
        "-c",
        type=str,
        dest="continue_from",
        help="Model checkpoint to continue the"
        + "training process from. Can be 'latest' or an int (for epoch number)",
    )

    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")

    args = parser.parse_args()
    return args


def save_results(log_path, batch_id, epoch, input_xyz, skel_xyz, skel_r):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()

    for i in range(batch_size):
        save_name_input = os.path.join(log_path, f"trn_{batch_id[i]}_input.ply")
        save_name_sphere = os.path.join(
            log_path, f"trn_{batch_id[i]}_sphere_{epoch}.obj"
        )
        save_name_center = os.path.join(
            log_path, f"trn_{batch_id[i]}_center_{epoch}.ply"
        )
        rw.save_ply_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_ply_points(skel_xyz_save[i], save_name_center)


def main(args):
    experiment_dir = args.exp_dir
    split_file = args.split
    continue_from = args.continue_from
    gpu = args.gpu

    # Setup the checkpoint and model eval dirs in exp_dir
    checkpt_dir = os.path.join(experiment_dir, workspace.checkpoint_subdir)
    eval_dir = os.path.join(experiment_dir, workspace.evaluation_subdir, "training")
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

    learning_rate = specs["LearningRate"]
    batch_size = specs["BatchSize"]
    to_normalize = specs["Normalize"]
    epochs_pretrain = specs["EpochsPreTrain"]
    epochs_skelpoint = specs["EpochsSkelPoint"]
    save_epoch = specs["SaveEvery"]
    point_num = specs["InputPointNum"]
    skelpoint_num = specs["SkelPointNum"]
    # Assume Training/Test split file (given as cmd line arg) will be present in the experiment dir
    pc_list_file = os.path.join(experiment_dir, split_file)

    # intialize network, optimizer, tensorboard
    model_skel = SkelPointNet(
        num_skel_points=skelpoint_num, input_channels=0, use_xyz=True
    )
    optimizer_skel = torch.optim.Adam(model_skel.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print("GPU Number:", torch.cuda.device_count(), "GPUs!")
        model_skel.cuda()
        model_skel.train(mode=True)
    else:
        print("No CUDA detected. Exiting....")
        return 0
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    print(f"stamp: {TIMESTAMP}")
    tb_writer = SummaryWriter(os.path.join(experiment_dir, "tb_logs"))

    # load data and train
    pc_list = rw.load_data_id(pc_list_file)
    train_data = PCDataset(pc_list, data_dir, point_num, to_normalize)
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    total_epochs = epochs_pretrain + epochs_skelpoint
    print(f"Pre-Training till {epochs_pretrain} epochs")
    print(f"Then skeletal point training till {total_epochs} epochs")
    print(f"Len dataloader: {len(train_loader)} | Len dataset: {len(train_data)}")

    if continue_from is not None:
        model_epoch = workspace.load_model_checkpoint(
            experiment_dir, continue_from, model_skel
        )
        optim_epoch = workspace.load_optimizer(
            experiment_dir, continue_from, optimizer_skel
        )
        if model_epoch != optim_epoch:
            raise RuntimeError(
                f"Checkpoint Epoch mismatch: model={model_epoch} vs optimizer={optim_epoch}"
            )
        starting_epoch = model_epoch + 1
        print(f"Resuming training from saved checkpoint:")
        print(f"{continue_from} checkpoint | {starting_epoch} epoch.\n")
    else:
        starting_epoch = 1

    print("Begin Training....")
    for epoch in tqdm(range(starting_epoch, total_epochs + 1)):
        loss_epoch = 0
        print(f"\n-----Epoch={epoch}-----")
        for _, batch_data in enumerate(tqdm(train_loader)):
            batch_id, batch_pc = batch_data
            batch_pc = batch_pc.cuda().float()

            optimizer_skel.zero_grad()
            skel_xyz, skel_r, _ = model_skel(batch_pc, compute_graph=False)
            ### pre-train skeletal point network
            if epoch <= epochs_pretrain:
                loss = model_skel.compute_loss_pre(batch_pc, skel_xyz)
                tb_loss_tag = "SkeletonPoint/loss_pre"
                tb_epoch = epoch
            #### train skeletal point network with geometric losses
            else:
                loss = model_skel.compute_loss(
                    batch_pc, skel_xyz, skel_r, None, 0.3, 0.4
                )
                tb_loss_tag = "SkeletalPoint/loss_skel"
                tb_epoch = epoch - epochs_pretrain + 1
            loss.backward()
            optimizer_skel.step()
            loss_epoch += loss.item()
            if epoch % save_epoch == 0:
                save_results(eval_dir, batch_id, epoch, batch_pc, skel_xyz, skel_r)

        loss_epoch /= len(train_loader)
        tb_writer.add_scalar(tb_loss_tag, loss_epoch, tb_epoch)
        workspace.save_latest(experiment_dir, epoch, model_skel, optimizer_skel)

        if epoch % save_epoch == 0:
            workspace.save_checkpoint(experiment_dir, epoch, model_skel, optimizer_skel)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
