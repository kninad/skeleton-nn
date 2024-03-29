import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff


def compute_pc_chamfer(gt_points, gen_points):
    """
    This functions computes the symmetric chamfer distance between two input point clouds.
    Input:
        gt_points: numpy array (N,3) containing the points of the ground truth shape
        gen_points: numpy array (M,3) containing the points inferred for the shape
    Returns:
        Scalar (float) value for the sum of one-way chamfer dists
    """
    kdtree_gt_pts = KDTree(gt_points)

    distances_to_gt, _ = kdtree_gt_pts.query(gen_points)
    chamfer_to_gt = np.mean(np.square(distances_to_gt))

    kdtree_gen_pts = KDTree(gen_points)
    distances_to_gen, _ = kdtree_gen_pts.query(gt_points)
    chamfer_to_gen = np.mean(np.square(distances_to_gen))

    return chamfer_to_gt + chamfer_to_gen


def compute_pc_haussdorff(pts_A, pts_B):
    """
    This functions computes the symmetric hausdorff distance between two input point clouds.
    Input:
        pts_A: numpy array (N,3) containing the points of the ground truth shape
        pts_B: numpy array (M,3) containing the points inferred for the shape
    Returns:
        Scalar (float) value for the symmetric hausdorff dist
    """
    return max(directed_hausdorff(pts_A, pts_B)[0], directed_hausdorff(pts_A, pts_B)[0])


def knn_with_batch(p1, p2, k, is_max=False):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :param is_max: k-nearest neighbors or k-farthest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    top_dist, k_nn = torch.topk(dist, k, dim=2, largest=is_max)

    return k_nn


def distance_map_with_batch(p1, p2):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :return: for each point in p1, returns the distances to all the points in p2; size[B,N,M]
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    return dist


def closest_distance_with_batch(p1, p2, is_sum=True):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice


def point2sphere_distance_with_batch(p1, p2):
    """
    :param p1: size[B,N,3]
    :param p2: size[B,M,4]
    :return: the distances from p1 to the closest spheres in p2
    """
    assert p1.size(0) == p2.size(0) and p1.size(2) == 3 and p2.size(2) == 4

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)
    p2_xyzr = p2
    p2 = p2_xyzr[:, :, :, 0:3]
    p2_r = p2_xyzr[:, :, :, 3]

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)
    # dist is of shape [B, N, M]
    min_dist, min_indice = torch.min(dist, dim=2)
    min_indice = torch.unsqueeze(min_indice, 2)
    min_dist = torch.unsqueeze(min_dist, 2)

    p2_min_r = torch.gather(p2_r, 2, min_indice)
    min_dist = min_dist - p2_min_r
    min_dist = torch.norm(min_dist, 2, dim=2)
    # Error with earlier code. We were not averaging with the
    # number of input points (N), instead just summing over.
    # Hence we first do mean and then sum
    dist_scalar = torch.sum(torch.mean(min_dist, dim=1))
    return dist_scalar


def sphere2point_distance_with_batch(p1, p2):
    """
    :param p1: size[B,N,4]
    :param p2: size[B,M,3]
    :return: the distances from sphere p1 to the closest points in p2
    """

    assert p1.size(0) == p2.size(0) and p1.size(2) == 4 and p2.size(2) == 3

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1_r = p1[:, :, :, 3]
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1_r = p1_r.transpose(1, 2)

    p1 = p1.transpose(1, 2)
    p1_xyzr = p1
    p1 = p1_xyzr[:, :, :, 0:3]

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    min_dist = torch.unsqueeze(min_dist, 2)
    min_dist = min_dist - p1_r
    min_dist = torch.norm(min_dist, 2, dim=2)
    # Error with earlier code. We were not averaging with the
    # number of input points (N), instead just summing over.
    # Hence we first do mean and then sum
    dist_scalar = torch.sum(torch.mean(min_dist, dim=1))
    return dist_scalar


def modified_sphere2point_distance_with_batch(p1, p2, topK=3):
    """
    :param p1: size[B,N,4]
    :param p2: size[B,M,3]
    :return: the distances from sphere p1 to the closest points in p2
    """

    assert p1.size(0) == p2.size(0) and p1.size(2) == 4 and p2.size(2) == 3

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1_r = p1[:, :, :, 3]
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1_r = p1_r.transpose(1, 2)

    p1 = p1.transpose(1, 2)
    p1_xyzr = p1
    p1 = p1_xyzr[:, :, :, 0:3]

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)
    # dist is a [B, N, M] shape tensor

    min_dist, _ = torch.topk(dist, k=topK, dim=2, largest=False)
    # Dont need the unsqueeze operation on dim=2 since torch.topK
    # will return a tensor of shape [B, N, K]
    min_dist = min_dist - p1_r
    min_dist = torch.norm(min_dist, 2, dim=2)
    dist_scalar = torch.sum(torch.mean(min_dist, dim=1))
    return dist_scalar


def closest_distance_np(p1, p2, is_sum=True):
    """
    :param p1: size[N, D], numpy array
    :param p2: size[M, D], numpy array
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    """

    p1 = torch.from_numpy(p1[None, :, :]).double()
    p2 = torch.from_numpy(p2[None, :, :]).double()

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=2)
    min_dist, min_indice = torch.min(dist, dim=1)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice
