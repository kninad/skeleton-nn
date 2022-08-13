import os
import sys
import json
import glob

import vtk
import pyvista as pv

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from SkelPointNet import SkelPointNet 
from DataUtil import Srep, HippocampiProcessedData, LeafletData
import FileRW as rw
import DistFunc as DF

sys.path.insert(0, "../")
import utils.misc as workspace


### Util Functions ###

def log_results_label(log_path, batch_id, input_xyz, skel_xyz, skel_r, label_xyz):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    label_xyz_save = label_xyz.detach().cpu().numpy()
    for i in range(batch_size):
        save_name_input = os.path.join(log_path, f"val_{batch_id[i]}_input.ply")
        save_name_sphere = os.path.join(log_path, f"val_{batch_id[i]}_sphere.obj")
        save_name_center = os.path.join(log_path, f"val_{batch_id[i]}_center.ply")
        save_name_label = os.path.join(log_path, f"val_{batch_id[i]}_label.ply")
        rw.save_ply_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_ply_points(skel_xyz_save[i], save_name_center)
        rw.save_ply_points(label_xyz_save[i], save_name_label)


def compute_metrics_skeletal(batch_id, label_xyz, skel_xyz):
    """
    label_xyz : the g.t srep points
    skel_xyz : predicted skeletal points
    """
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    label_xyz_save = label_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    cd = 0
    hd = 0
    for i in range(batch_size):
        cd += DF.compute_pc_chamfer(label_xyz_save[i], skel_xyz_save[i])
        hd += DF.compute_pc_haussdorff(label_xyz_save[i], skel_xyz_save[i])
    return cd, hd


def implied_boundary_single(skel_pts, spoke_dirs, radii):
    """
    Input:
        skel_pts : (N,3) numpy array of skeletal points
        spoke_dirs : (N,3) numpy array of spoke directions
        radii : (N,1) numpy array of radii for each skel pt
    Returns:
        bdry_pts : (N, 3) points -- implied boundary points
    """
    return skel_pts + spoke_dirs * radii


def compute_metrics_implied_boundary(batch_id, skel_xyz, skel_r, spoke_xyz, input_xyz):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    spoke_xyz_save = spoke_xyz.detach().cpu().numpy()
    cd = 0
    hd = 0
    for i in range(batch_size):
        pred_bdry = implied_boundary_single(
            skel_xyz_save[i], spoke_xyz_save[i], skel_r_save[i]
        )
        cd += DF.compute_pc_chamfer(input_xyz_save[i], pred_bdry)
        hd += DF.compute_pc_haussdorff(input_xyz_save[i], pred_bdry)
    return cd, hd


def get_srep_bdry_meta(meta_dict, index, count):
    srep_path = meta_dict["label_f"][index]
    scale = meta_dict["scale"][index].detach().cpu().numpy()
    offset = meta_dict["offset"][index].detach().cpu().numpy()
    srep_obj = Srep(srep_path)
    srep_bdry = srep_obj.get_implied_boundary_pts()
    idxs = np.random.randint(srep_bdry.shape[0], size=count)
    srep_bdry = srep_bdry[idxs, :]
    srep_bdry -= offset
    srep_bdry /= scale
    return np.array(srep_bdry)


def compute_metrics_bdry_diff(batch_id, batch_meta, skel_xyz, skel_r, spoke_xyz):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    spoke_xyz_save = spoke_xyz.detach().cpu().numpy()
    cd = 0
    hd = 0
    pred_count = skel_xyz.size()[1]
    for i in range(batch_size):
        pred_bdry = implied_boundary_single(
            skel_xyz_save[i], spoke_xyz_save[i], skel_r_save[i]
        )
        srep_bdry = get_srep_bdry_meta(batch_meta, i, pred_count)
        cd += DF.compute_pc_chamfer(srep_bdry, pred_bdry)
        hd += DF.compute_pc_haussdorff(srep_bdry, pred_bdry)
    return cd, hd


def compute_metrics_bdry_srep(batch_id, batch_meta, input_xyz, skel_xyz):
    batch_size = input_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    cd = 0
    hd = 0
    pred_count = skel_xyz.size()[1]
    for i in range(batch_size):
        srep_bdry = get_srep_bdry_meta(batch_meta, i, pred_count)
        cd += DF.compute_pc_chamfer(srep_bdry, input_xyz_save[i])
        hd += DF.compute_pc_haussdorff(srep_bdry, input_xyz_save[i])
    return cd, hd


### Core Test Script ###

def test_results(experiment_dir, eval_dataset, model, save_results=False, view_vtk=False, srep_res=False, if_xml=False):
    eval_save_dir = os.path.join(experiment_dir, workspace.evaluation_subdir, "eval")
    rw.check_and_create_dirs([eval_save_dir])

    data_loader = DataLoader(
        dataset=eval_dataset, batch_size=1, shuffle=False, drop_last=False
    )

    label_loss_cd = 0
    label_loss_hd = 0

    bdry_cd = 0
    bdry_hd = 0

    diff_cd = 0
    diff_hd = 0

    srep_bdry_cd = 0
    srep_bdry_hd = 0

    for _, batch_data in enumerate(tqdm(data_loader)):
        batch_id, batch_pc, batch_label, batch_meta = batch_data
        batch_pc = batch_pc.cuda().float()
        with torch.no_grad():
            skel_xyz, skel_r, spokes, *_ = model(batch_pc)

            # Cdist and Hdist between the input boundary points and
            # predicted boundary points
            cd_batch_bdry, hd_batch_bdry = compute_metrics_implied_boundary(
                batch_id, skel_xyz, skel_r, spokes, batch_pc
            )
            bdry_cd += cd_batch_bdry
            bdry_hd += hd_batch_bdry

            # Cdist and Hdist between the g.t srep points and
            # predicted skel points
            cd_batch_srep, hd_batch_srep = compute_metrics_skeletal(
                batch_id, batch_label, skel_xyz
            )
            label_loss_cd += cd_batch_srep
            label_loss_hd += hd_batch_srep

            if srep_res:
                # # Cdist and Hdist between srep bdry pts and
                # # predicted bdry points
                cd_batch_diff, hd_batch_diff = compute_metrics_bdry_diff(
                    batch_id, batch_meta, skel_xyz, skel_r, spokes
                )
                diff_cd += cd_batch_diff
                diff_hd += hd_batch_diff

                # # Cdist and Hdist between srep bdry pts and
                # # input boundary points
                cd_batch_srepbdry, hd_batch_srepbdry = compute_metrics_bdry_srep(
                    batch_id, batch_meta, batch_pc, skel_xyz
                )
                srep_bdry_cd += cd_batch_srepbdry
                srep_bdry_hd += hd_batch_srepbdry
            viz_save_dir = None
            if save_results:
                log_results_label(
                    eval_save_dir,
                    batch_id,
                    batch_pc,
                    skel_xyz,
                    skel_r,
                    batch_label
                )
                viz_save_dir = eval_save_dir
            if view_vtk:
                view_results_vtk(
                    batch_id,
                    batch_meta,
                    skel_xyz,
                    skel_r,
                    spokes,
                    if_xml,
                    viz_save_dir
                )

    N = len(eval_dataset)
    label_loss_cd /= N
    label_loss_hd /= N
    bdry_hd /= N
    bdry_cd /= N
    diff_cd /= N
    diff_hd /= N
    srep_bdry_cd /= N
    srep_bdry_hd /= N
    print("---------------")
    print("CDist and HDist of pred bdry with input bdry")
    print(bdry_cd, bdry_hd)
    print("---------------")
    print("CDist and HDist of srep bdry with input bdry")
    print(srep_bdry_cd, srep_bdry_hd)
    print("---------------")
    print("CDist and HDist of pred bdry with srep bdry")
    print(diff_cd, diff_hd)
    print("---------------")
    print("CDist and HDist of pred skel with g.t srep pts")
    print(label_loss_cd, label_loss_hd)


### VTK Vizualization Code ###

def view_results_vtk(batch_id, batch_meta, skel_xyz, skel_r, spoke_xyz, xml_data=False, save_dir=None):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    spoke_xyz_save = spoke_xyz.detach().cpu().numpy()
    for i in range(batch_size):
        scale = batch_meta['scale'][i].detach().cpu().numpy()
        offset = batch_meta['offset'][i].detach().cpu().numpy()
        input_f = batch_meta['input_f'][i]
        # get implied boundary, and obtain its vtk polyData form
        pred_bdry = implied_boundary_single(
            skel_xyz_save[i], spoke_xyz_save[i], skel_r_save[i]
        )
        tf_skel = skel_xyz_save[i] * scale + offset
        tf_bdry = pred_bdry * scale + offset
        output_skel = get_vtk_srep_mesh(tf_skel, tf_bdry)
        # read input mesh (.vtk file)
        if xml_data:
            reader = vtk.vtkXMLPolyDataReader()
        else:
            reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_f)
        reader.Update()
        input_mesh = reader.GetOutput()
        output_pts = pv.PolyData(tf_skel)
        if save_dir:
            pc_f = os.path.join(save_dir, f"{batch_id[i]}_pc")
            spoke_f = os.path.join(save_dir, f"{batch_id[i]}_spokes")
        else:
            pc_f, spoke_f = None, None
        view_vtk_pc(output_pts, input_mesh, pc_f)
        view_vtk_spokes(output_skel, input_mesh, spoke_f)


def get_vtk_srep_mesh(skel_pts, bdry_pts):
    '''
    Input
        skel_pts : (N, 3) skeleton points
        spoke_dirs : (N, 3) unit vectors for spoke direction
        radii : (N, 1) radius of medial spheres
    '''
    srep_poly = vtk.vtkPolyData()
    srep_pts = vtk.vtkPoints()
    srep_cells = vtk.vtkCellArray()

    for i in range(skel_pts.shape[0]):
        id_s = srep_pts.InsertNextPoint(skel_pts[i, :])
        id_b = srep_pts.InsertNextPoint(bdry_pts[i, :])

        tmp_spoke = vtk.vtkLine()
        tmp_spoke.GetPointIds().SetId(0, id_s)
        tmp_spoke.GetPointIds().SetId(1, id_b)
        srep_cells.InsertNextCell(tmp_spoke)

    srep_poly.SetPoints(srep_pts)
    srep_poly.SetLines(srep_cells)
    return srep_poly


def view_vtk_spokes(srep, mesh, save_f=None):
    plt = pv.Plotter()
    plt.add_mesh(mesh, color='white', opacity=0.2)
    plt.add_mesh(srep)
    if save_f:
        plt.export_vtkjs(save_f)
    else:
        plt.show()

def view_vtk_pc(srep_pts, mesh, save_f=None):
    plt = pv.Plotter()
    plt.add_mesh(mesh, color='white', opacity=0.2)
    plt.add_points(srep_pts, color='white', point_size=2)
    if save_f:
        plt.export_vtkjs(save_f)
    else:
        plt.show()


if __name__ == "__main__":

    # Test Code
    EXP_TYPE = "leaflet" # "hippocampi" or "leaflet"

    if EXP_TYPE == "hippocampi":
        EXP_NAME = "gt-full5000-pskel100-finetune_1010-hipp"
    else:
        EXP_NAME = "gt-full5000-pskel100-finetune_1010-leaf"

    experiment_dir = os.path.join("../experiments/", EXP_NAME)
    checkpoint = 'latest'
    with open(os.path.join(experiment_dir, "specs.json"), "r") as f:
        specs = json.load(f)
    
    point_num = specs["InputPointNum"]
    skelpoint_num = specs["SkelPointNum"]
    to_normalize = specs["Normalize"]
    gpu = "0"

    flag_sup = True  # supervision
    flags_vec = specs.get("FlagsVec", [])
    if not flags_vec:
        flags_vec = [0] * 4
    flags_boolvec = [x==1 for x in flags_vec]
    flag_spread, flag_radius, flag_medial, flag_spoke = flags_boolvec
    print(f"FlagVec: {flags_vec}")
    print(f"FLAGS: {flag_spread}, {flag_radius}, {flag_medial}, {flag_spoke}!!!")

    model_skel = SkelPointNet(
        num_skel_points=skelpoint_num, input_channels=0, use_xyz=True,
        flag_supervision=flag_sup,
        flag_spread=flag_spread,
        flag_radius=flag_radius,
        flag_medial=flag_medial,
        flag_spoke=flag_spoke
    )

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        print("GPU Number:", torch.cuda.device_count(), "GPUs!")
        model_skel.cuda()
        model_skel.eval()
    # Load the saved model
    model_epoch = workspace.load_model_checkpoint(
        experiment_dir, checkpoint, model_skel
    )
    print(f"Evaluating model on using checkpoint={checkpoint} and epoch={model_epoch}.")
    
    base_data_dir = specs["BaseDataSource"]
    if EXP_TYPE == "hippocampi":
        data_dir = os.path.join("../data/", "hippocampi_processed")
    elif EXP_TYPE == "leaflet":
        data_dir = os.path.join("../data/", "leaflet_sreps")

    # load data and evaluate
    if EXP_TYPE == "hippocampi":
        data_list = sorted(
            glob.glob(os.path.join(data_dir, "surfaces", "*_surf_SPHARM.vtk"))
        )
        label_list = sorted(glob.glob(os.path.join(data_dir, "sreps", "*.srep.json")))
    elif EXP_TYPE == "leaflet":
        case_dirs = sorted(os.listdir(data_dir))
        data_list = [os.path.join(data_dir, case, "warped_template.vtp") for case in case_dirs]
        label_list = [os.path.join(data_dir, case, "up_proc.vtp") for case in case_dirs]

    idx_end = int(len(data_list) * 0.9)
    data_list_eval = data_list[idx_end:]
    label_list_eval = label_list[idx_end:]

    if EXP_TYPE == "hippocampi":
        eval_data = HippocampiProcessedData(
            data_list_eval, label_list_eval, point_num, load_in_ram=True
        )
        use_xml_loader = False
    elif EXP_TYPE == "leaflet":
        eval_data = LeafletData(
            data_list_eval, label_list_eval, point_num, load_in_ram=True
        )
        use_xml_loader = True


    test_results(experiment_dir, eval_data, model_skel, 
        save_results=True, 
        view_vtk=True,
        srep_res=True,
        if_xml=use_xml_loader
    )
