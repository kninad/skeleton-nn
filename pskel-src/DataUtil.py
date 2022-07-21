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
        fpath = os.path.join(self.data_folder, self.data_id[i] + ".nrrd")
        data = reader.read(fpath)
        img, meta = reader.get_data(data)
        img = np.asarray(img, np.unint8)
        pts = self.image_to_point_cloud(img)
        return pts

    def image_to_point_cloud(self, image, threshold=0):
        # bounds is a numpy array with only the boundary voxels as 1 (True)
        bounds = find_boundaries(image)
        # Get voxel coordinates for the boundary voxels
        pointSet = np.argwhere(image > threshold)  # > 0 since binary image
        pointSet = np.array(pointSet, dtype=np.float32)
        pointSet /= bounds.shape  # Set the coords to be between [0, 1]
        if pointSet.shape[0] <= self.point_num:
            raise AssertionError("Low number of boundary points in image!")
        idxs = np.random.randint(pointSet.shape[0], size=self.point_num)
        pointSet = pointSet[idxs, :]
        pointSet = self.normalize(pointSet)
        return pointSet

    def normalize(self, pts):
        pts -= np.mean(pts, axis=0)
        pts /= np.max(np.linalg.norm(pts, axis=1))  # max dist
        return pts
