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
        meta = {
            "scale": data_scale,
            "offset": data_center
        }
        return index, data_pc, label_pc, meta

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
            self.data_pc, self.label_pc, self.meta_info = self.preprocess()

    def __getitem__(self, index):
        if self.load_in_ram:
            return index, self.data_pc[index], self.label_pc[index], self.meta_info[index]
        else:
            data, center, scale = self.load_data(index)
            label = self.load_label(index, center, scale)
            meta = {
                "scale": scale,
                "offset": center,
                "label_f": self.label_list[index],
                "input_f": self.data_list[index]
            }
            return index, data, label, meta

    def __len__(self):
        return len(self.data_list)

    def preprocess(self):
        data_samples = []
        label_samples = []
        meta_samples = []
        for idx in range(len(self.data_list)):
            data, center, scale = self.load_data(idx)
            label = self.load_label(idx, center, scale)
            meta = {
                "scale": scale,
                "offset": center,
                "label_f": self.label_list[idx],
                "input_f": self.data_list[idx]
            }
            data_samples.append(data)
            label_samples.append(label)
            meta_samples.append(meta)
        return data_samples, label_samples, meta_samples

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
            self.data_pc, self.label_pc, self.meta_info = self.preprocess()

    def __getitem__(self, index):
        if self.load_in_ram:
            return index, self.data_pc[index], self.label_pc[index], self.meta_info[index]
        else:
            data, center, scale = self.load_data(index)
            label = self.load_label(index, center, scale)
            meta = {
                "scale": scale,
                "offset": center,
                "label_f": self.label_list[index],
                "input_f": self.data_list[index]
            }
            return index, data, label, meta

    def __len__(self):
        return len(self.data_list)

    def preprocess(self):
        data_samples = []
        label_samples = []
        meta_samples = []
        for idx in range(len(self.data_list)):
            data, center, scale = self.load_data(idx)
            label = self.load_label(idx, center, scale)
            meta = {
                "scale": scale,
                "offset": center,
                "label_f": self.label_list[idx],
                "input_f": self.data_list[idx]
            }
            data_samples.append(data)
            label_samples.append(label)
            meta_samples.append(meta)
        return data_samples, label_samples, meta_samples

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


class Srep:

    def __init__(self, srep_path):
        if srep_path.endswith('.json'):
            skel_pts, up_spokes, dn_spokes, crest_pts, crest_spokes = self.extract_srep_json(srep_path)
        else:
            # Check if its a *.vtp file path instead of directory -- if so then extract the dir from it.
            if (not os.path.isdir(srep_path)) and srep_path.endswith('.vtp'):
                srep_dir = os.path.dirname(srep_path)
            else:
                srep_dir = srep_path
            skel_pts, up_spokes, dn_spokes, crest_pts, crest_spokes = self.extract_srep_vtp(srep_dir)
        
        self.all_points = np.concatenate([skel_pts, crest_pts], axis=0)
        self.skel_pts = skel_pts
        self.crest_pts = crest_pts
        self.up_spokes = up_spokes
        self.down_spokes = dn_spokes
        self.crest_spokes = crest_spokes

    def get_implied_boundary_pts(self):
        N = self.skel_pts.shape[0]
        M = self.crest_pts.shape[0]
        boundary_pts = np.zeros((2*N+M, 3))
        # from the skel points and up/down spokes
        for i in range(N):
            idx_up = i
            idx_dn = i + N
            boundary_pts[idx_up] = self.skel_pts[i] + self.up_spokes[i]
            boundary_pts[idx_dn] = self.skel_pts[i] + self.down_spokes[i]
        # from the crest points
        for i in range(M):
            idx_crest = 2*N + i
            boundary_pts[idx_crest] = self.crest_pts[i] + self.crest_spokes[i]
        return boundary_pts

    # vtp loading
    def _util_extract_indiv_srep_vtp(self, fpath):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(fpath)
        reader.Update()
        polydata = reader.GetOutput()
        points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        pointdata = polydata.GetPointData()
        directions = numpy_support.vtk_to_numpy(pointdata.GetArray("spokeDirection"))
        radii = numpy_support.vtk_to_numpy(pointdata.GetArray("spokeLength"))
        spokes = np.expand_dims(radii, axis=1) * directions
        return np.array(points), np.array(spokes), np.array(radii), np.array(directions)

    def extract_srep_vtp(self, data_dir):
        up_f = os.path.join(data_dir, "up_proc.vtp")
        down_f = os.path.join(data_dir, "down_proc.vtp")
        crest_f = os.path.join(data_dir, "crest_proc.vtp")
        p_up, up_spokes, r_up, d_up = self._util_extract_indiv_srep_vtp(up_f)
        p_dn, dn_spokes, r_dn, d_dn = self._util_extract_indiv_srep_vtp(down_f)
        p_cr, cr_spokes, r_cr, d_cr = self._util_extract_indiv_srep_vtp(crest_f)
        return p_up, up_spokes, dn_spokes, p_cr, cr_spokes


    # Json loading
    def _util_get_data_from_inner_list(self, inner_list):
        skel_points = []
        up_spokes = []
        dn_spokes = []
        crest_points = []
        crest_spokes = []
        for item in inner_list:
            spoke_pt = item['UpSpoke']['SkeletalPoint']['Value']
            up_dir = item['UpSpoke']['Direction']['Value']
            dn_dir = item['DownSpoke']['Direction']['Value']
            skel_points.append(spoke_pt)
            up_spokes.append(up_dir)
            dn_spokes.append(dn_dir)
            if 'CrestSpoke' in item:
                crest_pt = item['CrestSpoke']['SkeletalPoint']['Value']
                crest_dir = item['CrestSpoke']['Direction']['Value']
                crest_points.append(crest_pt)
                crest_spokes.append(crest_dir)
        return skel_points, up_spokes, dn_spokes, crest_points, crest_spokes

    def _util_get_data_from_skeleton(self, skeleton_list):
        all_skel_points = []
        all_up_spokes = []
        all_dn_spokes = []
        all_crest_points = []
        all_crest_spokes = []
        # skel_points, up_spokes, dn_spokes, crest_points, crest_spokes
        all_data = [self._util_get_data_from_inner_list(inner_list) 
                        for inner_list in skeleton_list]
        for skel_points, up_spokes, dn_spokes, crest_points, crest_spokes in all_data:
            all_skel_points += skel_points
            all_up_spokes += up_spokes
            all_dn_spokes += dn_spokes
            all_crest_points += crest_points
            all_crest_spokes += crest_spokes
        return all_skel_points, all_up_spokes, all_dn_spokes, all_crest_points, all_crest_spokes

    def extract_srep_json(self, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        srepdata = data['EllipticalSRep']
        skeldata = srepdata['Skeleton']
        srep_data = self._util_get_data_from_skeleton(skeldata)
        skel_pts = np.array(srep_data[0])
        up_spokes = np.array(srep_data[1])
        dn_spokes = np.array(srep_data[2])
        crest_pts = np.array(srep_data[3])
        crest_spokes = np.array(srep_data[4])
        return skel_pts, up_spokes, dn_spokes, crest_pts, crest_spokes
