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


# import config as conf


def parse_arguments():
    parser = argparse.ArgumentParser(description='Point2Skeleton')

    parser.add_argument('--exp_dir', '-e', type=str,
                        default="../experiments/init_run/", help="Path to hyperparams dir.")

    parser.add_argument('--continue', '-c', type=str,
                        dest="continue_from",
                        help="Model checkpoint to continue the"
                        + "training process from. Can be 'latest' or an int (for epoch number)")

    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')





    parser.add_argument('--pc_list_file', type=str, default='../data/data-split/all-train.txt',
                        help='file of the names of the point clouds')

    parser.add_argument('--data_root', type=str, default='../data/pointclouds/',
                        help='root directory of all the data')

    parser.add_argument('--point_num', type=int, default=2000, help='input point number')

    parser.add_argument('--skelpoint_num', type=int, default=100, help='output skeletal point number')

    parser.add_argument('--save_net_path', type=str, default='../training-weights/',
                        help='directory to save the network parameters')
    parser.add_argument('--save_net_iter', type=int, default=1000,
                        help='frequency to save the network parameters (number of iteration)')
    parser.add_argument('--save_log_path', type=str, default='../tensorboard/',
                        help='directory to save the training log (tensorboard)')
    parser.add_argument('--save_result_path', type=str, default='../log/',
                        help='directory to save the temporary results during training')
    parser.add_argument('--save_result_iter', type=int, default=1000,
                        help='frequency to save the intermediate results (number of iteration)')

    args = parser.parse_args()
    return args


def save_results(log_path, batch_id, epoch, input_xyz, skel_xyz, skel_r):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()

    for i in range(batch_size):
        save_name_input = os.path.join(log_path, f"trn_{batch_id[i]}_input.off")
        save_name_sphere = os.path.join(log_path, f"trn_{batch_id[i]}_sphere_{epoch}.obj")
        save_name_center = os.path.join(log_path, f"trn_{batch_id}_center_{epoch}.off")
        rw.save_off_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_off_points(skel_xyz_save[i], save_name_center)


def main(args):    
    experiment_dir = args.exp_dir
    continue_from = args.continue_from
    gpu = args.gpu

    # Setup the checkpoint and model eval dirs in exp_dir
    checkpt_dir = os.path.join(experiment_dir, workspace.checkpoint_subdir)
    eval_dir = os.path.join(experiment_dir, workspace.evaluation_subdir)
    rw.check_and_create_dirs([checkpt_dir, eval_dir])

    with open(os.path.join(experiment_dir, "specs.json"), "r") as f:
        specs = json.load(f)
    
    data_dir = specs["DataSource"]
    if not os.path.isdir(data_dir):
        print(
            f"The provided data dir in specs.json is invalid! {data_dir} is not found on this system.")
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
    # Assume Training split file will be present in the experiment dir as "train_split.txt"
    pc_list_file = os.path.join(experiment_dir, "train_split.txt")


    #intialize network, optimizer, tensorboard
    model_skel = SkelPointNet(num_skel_points=skelpoint_num, input_channels=0, use_xyz=True)
    optimizer_skel = torch.optim.Adam(model_skel.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print("GPU Number:", torch.cuda.device_count(), "GPUs!")
        model_skel.cuda()
        model_skel.train(mode=True)
    else:
        print("No CUDA detected. Exiting....")
        return 0
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    print(f"stamp: {TIMESTAMP}")
    tb_writer = SummaryWriter(os.path.join(experiment_dir, f"tb_logs_{TIMESTAMP}"))

    #load data and train
    pc_list = rw.load_data_id(pc_list_file)
    train_data = PCDataset(pc_list, data_dir, point_num, to_normalize)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    total_epochs = epochs_pretrain + epochs_skelpoint
    print(f"Pre-Training till {epochs_pretrain} epochs")
    print(f"Then skeletal point training till {total_epochs} epochs")
    print(f"Len dataloader: {len(train_loader)} | Len dataset: {len(train_data)}")
    
    if continue_from is not None:
        model_epoch = workspace.load_model_checkpoint(
            experiment_dir, continue_from, model_skel)
        optim_epoch = workspace.load_optimizer(
            experiment_dir, continue_from, optimizer_skel)
        if model_epoch != optim_epoch:
            raise RuntimeError(
                f"Checkpoint Epoch mismatch: model={model_epoch} vs optimizer={optim_epoch}")
        starting_epoch = model_epoch + 1
        print(f"Resuming training from saved checkpoint:") 
        print(f"{continue_from} checkpoint | {starting_epoch} epoch.\n")        
    else:
        starting_epoch = 1
    
    print("Begin Training....")
    for epoch in range(starting_epoch, total_epochs+1):
        loss_epoch = 0
        print(f"-----Epoch={epoch}-----")
        for _, batch_data in enumerate(tqdm(train_loader)):
            # print('epoch, iter:', epoch, iter)
            batch_id, batch_pc = batch_data
            batch_pc = batch_pc.cuda().float()
            
            optimizer_skel.zero_grad()
            skel_xyz, skel_r, _ = model_skel(batch_pc, compute_graph=False)           
            if epoch <= epochs_pretrain: ### pre-train skeletal point network
                loss = model_skel.compute_loss_pre(batch_pc, skel_xyz)
                tb_loss_tag = 'SkeletonPoint/loss_pre'
                tb_epoch = epoch
            else: # train skeletal point network with geometric losses
                loss = model_skel.compute_loss(batch_pc, skel_xyz, skel_r, None, 0.3, 0.4)
                tb_loss_tag = 'SkeletalPoint/loss_skel'
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
    #parse arguments
    args = parse_arguments()
    main(args)
        
