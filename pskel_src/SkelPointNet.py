import torch
import torch.nn as nn
import numpy as np
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG
import torch.nn.functional as F
import copy
import DistFunc as DF


class SkelPointNet(nn.Module):
    def __init__(
        self,
        num_skel_points,
        input_channels=3,
        use_xyz=True,
        flag_supervision=False,
        flag_spread=False,
        flag_radius=False,
        flag_medial=False,
        flag_spoke=False,
    ):

        super(SkelPointNet, self).__init__()
        self.point_dim = 3
        self.num_skel_points = num_skel_points
        self.input_channels = input_channels

        self.flag_supervision = flag_supervision
        self.flag_spread = flag_spread
        self.flag_radius = flag_radius
        self.flag_medial = flag_medial
        self.flag_spoke = flag_spoke

        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[input_channels, 16, 16, 32], [input_channels, 16, 16, 32]],
                use_xyz=use_xyz,
            )
        )

        input_channels = 32 + 32
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=768,
                radii=[0.2, 0.4],
                nsamples=[32, 64],
                mlps=[[input_channels, 32, 32, 64], [input_channels, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )

        input_channels = 64 + 64
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.4, 0.6],
                nsamples=[32, 64],
                mlps=[[input_channels, 64, 64, 128], [input_channels, 64, 64, 128]],
                use_xyz=use_xyz,
            )
        )

        input_channels = 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.6, 0.8],
                nsamples=[64, 128],
                mlps=[[input_channels, 128, 128, 256], [input_channels, 128, 128, 256]],
                use_xyz=use_xyz,
            )
        )

        input_channels = 256 + 256
        cvx_weights_modules = []

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(
            nn.Conv1d(in_channels=input_channels, out_channels=384, kernel_size=1)
        )
        cvx_weights_modules.append(nn.BatchNorm1d(384))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(
            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=1)
        )
        cvx_weights_modules.append(nn.BatchNorm1d(256))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)
        )
        cvx_weights_modules.append(nn.BatchNorm1d(256))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(nn.Dropout(0.2))
        cvx_weights_modules.append(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        )
        cvx_weights_modules.append(nn.BatchNorm1d(128))
        cvx_weights_modules.append(nn.ReLU(inplace=True))

        cvx_weights_modules.append(
            nn.Conv1d(in_channels=128, out_channels=self.num_skel_points, kernel_size=1)
        )
        cvx_weights_modules.append(nn.BatchNorm1d(self.num_skel_points))
        cvx_weights_modules.append(nn.Softmax(dim=2))

        self.cvx_weights_mlp = nn.Sequential(*cvx_weights_modules)

    def compute_spread_loss(self, skel_xyz):
        """
        Input:
            skel_xyz : Shape [B, N, 3]
        
        cdist computes a [B,N,N] matrix and we avg over each batch index (dim=(1,2))
        and then sum over the avgs for each batch input.
        Negative size to encourage higher avg pairwise distance (more spread)
        """
        if self.flag_spread:
            return -torch.sum(torch.mean(torch.cdist(skel_xyz, skel_xyz), dim=(1, 2)))
        else:
            return 0

    def compute_supervised_loss(self, srep_xyz, skel_xyz):
        if self.flag_supervision and (srep_xyz is None):
            raise AssertionError(
                "Either flag_supervision condition mis-specified or null g.t srep tensor!"
            )
        if not self.flag_supervision:
            return 0
        srep_pnum = float(srep_xyz.size()[1])
        skel_pnum = float(skel_xyz.size()[1])
        cd_srep_to_skel = DF.closest_distance_with_batch(srep_xyz, skel_xyz) / srep_pnum
        cd_skel_to_srep = DF.closest_distance_with_batch(skel_xyz, srep_xyz) / skel_pnum
        return cd_skel_to_srep + cd_srep_to_skel

    def compute_loss(
        self, shape_xyz, skel_xyz, skel_radius, spokes, w1, w2, w3, srep_xyz=None,
    ):
        skel_pnum = float(skel_xyz.size()[1])

        # sampling loss
        loss_sample = self.get_sampling_loss(shape_xyz, skel_xyz, skel_radius, spokes)

        # point2sphere loss
        loss_point2sphere = self.get_point2sphere_loss(shape_xyz, skel_xyz, skel_radius)

        # radius loss
        loss_radius = -torch.sum(skel_radius) / skel_pnum

        # supervision (returns 0 if flag_supervision is False)
        loss_gt = self.compute_supervised_loss(srep_xyz, skel_xyz)

        # skel_pts inter distance loss (returns 0 if flag_spread is False)
        loss_spread = self.compute_spread_loss(skel_xyz)

        # Final loss combination
        final_loss = (
            loss_gt
            + loss_sample
            + loss_point2sphere * w1
            + loss_radius * w2
            + loss_spread * w3
        )
        return final_loss

    def get_point2sphere_loss(self, shape_xyz, skel_xyz, skel_radius, topK=3):
        shape_pnum = float(shape_xyz.size()[1])
        skel_pnum = float(skel_xyz.size()[1])
        skel_xyzr = torch.cat((skel_xyz, skel_radius), 2)
        cd_point2pshere1 = (
            DF.point2sphere_distance_with_batch(shape_xyz, skel_xyzr) / shape_pnum
        )
        if self.flag_medial:
            cd_point2sphere2 = (
                DF.modified_sphere2point_distance_with_batch(skel_xyzr, shape_xyz, topK)
                / skel_pnum
            )
        else:
            cd_point2sphere2 = (
                DF.sphere2point_distance_with_batch(skel_xyzr, shape_xyz) / skel_pnum
            )
        loss_point2sphere = cd_point2pshere1 + cd_point2sphere2
        return loss_point2sphere

    def get_sampling_loss(self, shape_xyz, skel_xyz, skel_radius, spokes):
        if (spokes is None) and self.flag_spoke:
            raise AssertionError(
                "Either flag_spoke condition mis-specified or null spokes tensor!"
            )

        bn = skel_xyz.size()[0]
        shape_pnum = float(shape_xyz.size()[1])
        skel_pnum = float(skel_xyz.size()[1])

        # sampling loss
        if self.flag_spoke:
            e = 1.0  # we will use the spoke for the direction vector
        else:
            e = 0.57735027
        sample_directions = torch.tensor(
            [
                [e, e, e],
                [e, e, -e],
                [e, -e, e],
                [e, -e, -e],
                [-e, e, e],
                [-e, e, -e],
                [-e, -e, e],
                [-e, -e, -e],
            ]
        )
        sample_directions = torch.unsqueeze(sample_directions, 0)
        sample_directions = sample_directions.repeat(bn, int(skel_pnum), 1).cuda()
        if self.flag_spoke:
            sample_directions = sample_directions * torch.repeat_interleave(
                spokes, 8, dim=1
            )
        sample_centers = torch.repeat_interleave(skel_xyz, 8, dim=1)
        sample_radius = torch.repeat_interleave(skel_radius, 8, dim=1)
        sample_xyz = sample_centers + sample_radius * sample_directions

        cd_sample1 = DF.closest_distance_with_batch(sample_xyz, shape_xyz) / (
            skel_pnum * 8
        )
        cd_sample2 = DF.closest_distance_with_batch(shape_xyz, sample_xyz) / (
            shape_pnum
        )
        loss_sample = cd_sample1 + cd_sample2
        return loss_sample

    def get_smoothness_loss(self, skel_xyz, A, k=6):

        bn, pn, p_dim = skel_xyz.size()[0], skel_xyz.size()[1], skel_xyz.size()[2]

        if A is None:
            knn_min = DF.knn_with_batch(skel_xyz, skel_xyz, k, is_max=False)
            A = torch.zeros((bn, pn, pn)).float().cuda()
            for i in range(bn):
                for j in range(pn):
                    A[i, j, knn_min[i, j, 1:k]] = 1
                    A[i, knn_min[i, j, 1:k], j] = 1

        neighbor_num = torch.sum(A, dim=2, keepdim=True)
        neighbor_num = neighbor_num.repeat(1, 1, p_dim)

        dist_sum = torch.bmm(A, skel_xyz)
        dist_avg = torch.div(dist_sum, neighbor_num)

        lap = skel_xyz - dist_avg
        lap = torch.norm(lap, 2, dim=2)
        loss_smooth = torch.sum(lap)

        return loss_smooth

    def compute_loss_pre(self, shape_xyz, skel_xyz):

        cd1 = DF.closest_distance_with_batch(shape_xyz, skel_xyz)
        cd2 = DF.closest_distance_with_batch(skel_xyz, shape_xyz)
        loss_cd = cd1 + cd2
        loss_cd = loss_cd * 0.0001

        return loss_cd

    def init_graph(self, shape_xyz, skel_xyz, valid_k=8):

        bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]

        knn_skel = DF.knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)
        knn_sp2sk = DF.knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)

        A = torch.zeros((bn, pn, pn)).float().cuda()

        # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
        A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
        A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1

        # initialize A with topology prior
        A[
            torch.arange(bn)[:, None, None],
            torch.arange(pn)[None, :, None],
            knn_skel[:, :, 1:2],
        ] = 1
        A[
            torch.arange(bn)[:, None, None],
            knn_skel[:, :, 1:2],
            torch.arange(pn)[None, :, None],
        ] = 1

        # valid mask: known existing links + knn links
        valid_mask = copy.deepcopy(A)
        valid_mask[
            torch.arange(bn)[:, None, None],
            torch.arange(pn)[None, :, None],
            knn_skel[:, :, 1:valid_k],
        ] = 1
        valid_mask[
            torch.arange(bn)[:, None, None],
            knn_skel[:, :, 1:valid_k],
            torch.arange(pn)[None, :, None],
        ] = 1

        # known mask: known existing links + known absent links, used as the mask to compute binary loss
        known_mask = copy.deepcopy(A)
        known_indice = list(range(valid_k, pn))
        known_mask[
            torch.arange(bn)[:, None, None],
            torch.arange(pn)[None, :, None],
            knn_skel[:, :, known_indice],
        ] = 1
        known_mask[
            torch.arange(bn)[:, None, None],
            knn_skel[:, :, known_indice],
            torch.arange(pn)[None, :, None],
        ] = 1

        return A, valid_mask, known_mask

    def split_point_feature(self, pc):

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, input_pc):

        input_pc = input_pc.cuda()
        xyz, features = self.split_point_feature(input_pc)

        # obtain the sampled points and contextural features
        for pointnet_module in self.SA_modules:
            xyz, features = pointnet_module(xyz, features)
        sample_xyz, context_features = xyz, features

        # convex combinational weights
        weights = self.cvx_weights_mlp(context_features)

        # skeletal points: shape [B, N, 3]
        skel_xyz = torch.sum(weights[:, :, :, None] * sample_xyz[:, None, :, :], dim=2)

        # surface features
        shape_cmb_features = torch.sum(
            weights[:, None, :, :] * context_features[:, :, None, :], dim=3
        )
        shape_cmb_features = shape_cmb_features.transpose(1, 2)

        # radii
        if self.flag_radius:
            dist = torch.cdist(skel_xyz, sample_xyz)  # shape [B, N, M]
            skel_r = torch.sum(weights * dist, dim=2).unsqueeze(2)
        else:
            min_dists, min_indices = DF.closest_distance_with_batch(
                sample_xyz, skel_xyz, is_sum=False
            )
            skel_r = torch.sum(
                weights[:, :, :, None] * min_dists[:, None, :, None], dim=2
            )

        spokes = self._get_spokes(weights, skel_xyz, sample_xyz, topK=1)

        return skel_xyz, skel_r, spokes, shape_cmb_features, weights, sample_xyz

    def _get_spokes(self, weights, skel_xyz, sample_xyz, topK=2):
        topK_idxs = torch.argsort(weights, axis=2, descending=True)[
            :, :, :topK
        ]  # shape [B, N, K]
        weit_idxs = torch.gather(weights, dim=2, index=topK_idxs)  # shape [B, N, K]
        # pairwise difference vector (spoke candidates)
        diff = -1 * (
            skel_xyz.unsqueeze(2) - sample_xyz.unsqueeze(1)
        )  # shape [B, N, M, 3]
        B, N, M = diff.shape[:-1]  # shape [B, N, K, 3]

        res = torch.empty(B, N, topK, 3, device="cuda:0")
        for i in range(3):
            res[:, :, :, i] = torch.gather(diff[:, :, :, i], dim=2, index=topK_idxs)

        weighted_diff = weit_idxs.unsqueeze(3) * res  # shape [B, N, K, 3]
        spokes = torch.sum(weighted_diff, axis=2)
        spokes = F.normalize(spokes, p=2, dim=2)
        return spokes
