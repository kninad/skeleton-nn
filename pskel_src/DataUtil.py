import os
import numpy as np
from torch.utils.data import Dataset
from monai.data import image_reader
from skimage.segmentation import find_boundaries
import json
import vtk
from vtk.util import numpy_support
import FileRW as rw


class PCDataset(Dataset):
    def __init__(self, data_list, data_folder, point_num, normalize=False):
        self.data_id = data_list
        self.data_folder = data_folder
        self.point_num = point_num
        self.to_normalize = normalize

    def __getitem__(self, index):
        fpath = os.path.join(self.data_folder, self.data_id[index] + ".ply")
        data_pc, _, _ = rw.load_ply_points(fpath, self.point_num, self.to_normalize)
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
        data_pc, data_center, data_scale = rw.load_ply_points(fpath, self.point_num, self.to_normalize)
        label_fpath = os.path.join(self.label_folder, self.label_id[index] + ".ply")
        # Use the center and scale used to normalize data to also normalize the label point cloud
        # Ensures that the both data and label are in the same coordinate system
        label_pc, _, _ = rw.load_ply_points(label_fpath, center=data_center, scale=data_scale)
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


class HippocampiProcessedData(Dataset):
    def __init__(self, data_list, label_list, point_num, load_in_ram=True):
        self.data_list = data_list
        self.label_list = label_list
        if len(data_list) != len(label_list):
            raise AssertionError("Data and Label lists dont have same lengths.")
        self.point_num = point_num
        self.load_in_ram = load_in_ram
        if load_in_ram:
            self.data_pc, self.label_pc = self.preprocess()

    def __getitem__(self, index):
        if self.load_in_ram:
            return index, self.data_pc[index], self.label_pc[index]
        else:
            data, center, scale = self.load_data(index)
            label = self.load_label(index, center, scale)
            return index, data, label

    def __len__(self):
        return len(self.data_list)

    def preprocess(self):
        data_samples = []
        label_samples = []
        for idx in range(len(self.data_list)):
            data, center, scale = self.load_data(idx)
            label = self.load_label(idx, center, scale)
            data_samples.append(data)
            label_samples.append(label)
        return data_samples, label_samples

    def load_data(self, index):
        fpath = self.data_list[index]
        surf_reader = vtk.vtkPolyDataReader()
        surf_reader.SetFileName(fpath)
        surf_reader.Update()
        surf = surf_reader.GetOutput()
        surf_pts = numpy_support.vtk_to_numpy(surf.GetPoints().GetData())
        pointSet = np.array(surf_pts)        
        # Randomly sample desired number of points
        if pointSet.shape[0] <= self.point_num:
            raise AssertionError("Low number of boundary points in image!")
        idxs = np.random.randint(pointSet.shape[0], size=self.point_num)
        pointSet = pointSet[idxs, :]
        # Normalize the pointSet to be zero mean and inside unit sphere
        center = np.mean(pointSet, axis=0)
        pointSet -= center
        max_dist = np.max(np.linalg.norm(pointSet, axis=1))
        pointSet /= max_dist
        return pointSet, center, max_dist
    
    def load_label(self, index, center, scale):
        fpath = self.label_list[index]
        with open(fpath, "r") as f:
            data = json.load(f)
        srepdata = data['EllipticalSRep']
        skeldata = srepdata['Skeleton']
        all_points = self._util_get_points_from_skeleton(skeldata)
        pointSet = np.array(all_points)
        pointSet -= center
        pointSet /= scale
        return pointSet

    def _util_get_points_from_inner_list(self, inner_list):
        points = []
        for item in inner_list:
            spoke_pt = item['UpSpoke']['SkeletalPoint']['Value']
            points.append(spoke_pt)
            if 'CrestSpoke' in item:
                crest_pt = item['CrestSpoke']['SkeletalPoint']['Value']
                points.append(crest_pt)
        return points

    def _util_get_points_from_skeleton(self, skeleton_list):
        all_points = []
        for inner_list in skeleton_list:
            all_points += self._util_get_points_from_inner_list(inner_list)
        return all_points


class LeafletData(Dataset):
    def __init__(self, data_list, label_list, point_num, load_in_ram=True):
        self.data_list = data_list
        self.label_list = label_list
        if len(data_list) != len(label_list):
            raise AssertionError("Data and Label lists dont have same lengths.")
        self.point_num = point_num
        self.load_in_ram = load_in_ram
        if load_in_ram:
            self.data_pc, self.label_pc = self.preprocess()

    def __getitem__(self, index):
        if self.load_in_ram:
            return index, self.data_pc[index], self.label_pc[index]
        else:
            data, center, scale = self.load_data(index)
            label = self.load_label(index, center, scale)
            return index, data, label

    def __len__(self):
        return len(self.data_list)

    def preprocess(self):
        data_samples = []
        label_samples = []
        for idx in range(len(self.data_list)):
            data, center, scale = self.load_data(idx)
            label = self.load_label(idx, center, scale)
            data_samples.append(data)
            label_samples.append(label)
        return data_samples, label_samples

    def load_data(self, index):
        fpath = self.data_list[index]
        surf_reader = vtk.vtkXMLPolyDataReader()
        surf_reader.SetFileName(fpath)
        surf_reader.Update()
        surf = surf_reader.GetOutput()
        surf_pts = numpy_support.vtk_to_numpy(surf.GetPoints().GetData())
        pointSet = np.array(surf_pts)        
        # Randomly sample desired number of points
        if pointSet.shape[0] <= self.point_num:
            raise AssertionError("Low number of surface points in org point cloud!")
        idxs = np.random.randint(pointSet.shape[0], size=self.point_num)
        pointSet = pointSet[idxs, :]
        # Normalize the pointSet to be zero mean and inside unit sphere
        center = np.mean(pointSet, axis=0)
        pointSet -= center
        max_dist = np.max(np.linalg.norm(pointSet, axis=1))
        pointSet /= max_dist
        return pointSet, center, max_dist
    
    def load_label(self, index, center, scale):
        fpath = self.label_list[index]
        srep_reader = vtk.vtkXMLPolyDataReader()
        srep_reader.SetFileName(fpath)
        srep_reader.Update()
        srep = srep_reader.GetOutput()
        srep_pts = numpy_support.vtk_to_numpy(srep.GetPoints().GetData())
        pointSet = np.array(srep_pts) 
        pointSet -= center
        pointSet /= scale
        return pointSet
