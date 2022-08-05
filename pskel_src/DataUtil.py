import os
import numpy as np
from torch.utils.data import Dataset
from monai.data import image_reader
from skimage.segmentation import find_boundaries
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


class EllipsoidPcDataset(Dataset):
    def __init__(self, data_list, label_list, data_folder, label_folder, point_num, normalize=False):
        self.data_id = data_list
        self.label_id = label_list
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.point_num = point_num
        self.to_normalize = normalize

    def __getitem__(self, index):
        fpath = os.path.join(self.data_folder, self.data_id[index] + ".ply")
        data_pc = rw.load_ply_points(fpath, self.point_num, self.to_normalize)
        label_fpath = os.path.join(self.label_folder, self.label_id[index] + ".ply")
        label_pc = rw.load_ply_points(label_fpath, self.point_num, self.to_normalize)
        return index, data_pc, label_pc

    def __len__(self):
        return len(self.data_id)


class TestBinaryImageData(Dataset):
    def __init__(self, data_list, data_folder, point_num, load_in_ram=False):
        self.data_id = data_list
        self.data_folder = data_folder
        self.point_num = point_num
        self.load_in_ram = load_in_ram
        if load_in_ram:
            self.point_clouds = self.preprocess()

    def __getitem__(self, index):
        if self.load_in_ram:
            return index, self.point_clouds[index]
        else:
            return index, self.load_data(index)

    def __len__(self):
        return len(self.data_id)

    def preprocess(self):
        self.point_clouds = [self.load_data(idx) for idx in range(len(self.data_id))]

    def load_data(self, index):
        reader = image_reader.ITKReader()
        fpath = os.path.join(self.data_folder, self.data_id[index])
        data = reader.read(fpath)
        img, meta = reader.get_data(data)
        img = np.asarray(img, dtype=np.uint8)
        # Find boundaries and convert to boundary voxel indices [0, 1] form
        bounds = find_boundaries(img)
        pointSet = np.argwhere(bounds > 0)
        pointSet = np.array(pointSet, dtype=np.float32)
        pointSet /= bounds.shape
        # Randomly sample desired number of points
        if pointSet.shape[0] <= self.point_num:
            raise AssertionError("Low number of boundary points in image!")
        idxs = np.random.randint(pointSet.shape[0], size=self.point_num)
        pointSet = pointSet[idxs, :]
        # Normalize the pointSet to be zero mean and inside unit sphere
        pointSet -= np.mean(pointSet, axis=0)
        max_dist = np.max(np.linalg.norm(pointSet, axis=1))
        pointSet /= max_dist
        return pointSet
