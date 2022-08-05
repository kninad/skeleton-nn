import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import torch
import math
import FileRW as rw


class PCDataset(Dataset):
    def __init__(self, data_list, data_folder, point_num, normalize=False):
        self.data_id = data_list
        self.data_folder = data_folder
        self.point_num = point_num
        self.to_normalize = normalize

    def __getitem__(self, index):
        fpath = os.path.join(self.data_folder, self.data_id[index] + ".ply")
        data_pc = rw.load_ply_points(fpath, self.point_num, self.to_normalize)
        return index, data_pc

    def __len__(self):
        return len(self.data_id)
