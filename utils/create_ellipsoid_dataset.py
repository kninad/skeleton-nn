import os
import json
import argparse

from tqdm import tqdm
import numpy as np
import itk
import vtk
from itkwidgets import view
from vtk.util import numpy_support


def center_and_align(surf_pts, srep_pts):
    # Center surface and srep at the origin
    surf_center = np.mean(surf_pts, axis=0)
    surf_pts_centered = surf_pts - surf_center

    srep_pts_centered = srep_pts - surf_center

    # Axis align the templates
    u, s, v = np.linalg.svd(surf_pts_centered.transpose() @ surf_pts_centered)
    R = v.transpose() @ u.transpose()

    surf_pts_align = np.zeros(surf_pts.shape)
    for i in range(surf_pts.shape[0]):
        surf_pts_align[i, :] = u.transpose() @ surf_pts_centered[i, :]

    srep_pts_align = np.zeros(srep_pts.shape)
    for i in range(srep_pts.shape[0]):
        srep_pts_align[i, :] = u.transpose() @ srep_pts_centered[i, :]

    return surf_pts_align, srep_pts_align


def scale_along_axes(surf_pts, srep_pts):
    # Apply scaling
    # Expects the surf_pts and srep_pts to be aligned and centered.
    xscale = np.random.normal(1.0, 0.15)
    yscale = np.random.normal(1.0, 0.15)
    zscale = np.random.normal(1.0, 0.15)
    surf_pts_scaled = np.zeros(surf_pts.shape)
    surf_pts_scaled[:, 0] = surf_pts[:, 0]*xscale
    surf_pts_scaled[:, 1] = surf_pts[:, 1]*yscale
    surf_pts_scaled[:, 2] = surf_pts[:, 2]*zscale

    srep_pts_scaled = np.zeros(srep_pts.shape)
    srep_pts_scaled[:, 0] = srep_pts[:, 0]*xscale
    srep_pts_scaled[:, 1] = srep_pts[:, 1]*yscale
    srep_pts_scaled[:, 2] = srep_pts[:, 2]*zscale

    axis_scales = (xscale, yscale, zscale)
    return surf_pts_scaled, srep_pts_scaled, axis_scales


def bend_and_twist(surf_pts_scaled, srep_pts_scaled):
    # Apply bending/twisting
    th = np.random.normal(np.pi/6, np.pi/12)
    ph = np.random.normal(np.pi/3, np.pi/8)

    xmax = np.max(surf_pts_scaled[:, 0])
    xmin = np.min(surf_pts_scaled[:, 0])

    surf_pts_bt = np.zeros(surf_pts_scaled.shape)
    for i in range(surf_pts_scaled.shape[0]):
        if surf_pts_scaled[i, 0] > 0:
            frac = surf_pts_scaled[i, 0] / xmax
        else:
            frac = np.abs(surf_pts_scaled[i, 0]) / xmin

        surf_pts_bt[i, 0] = (
            surf_pts_scaled[i, 0]*np.cos(frac*th) - surf_pts_scaled[i, 2]*np.sin(frac*th))

        surf_pts_bt[i, 1] = (
            surf_pts_scaled[i, 1]*np.cos(frac*ph) - surf_pts_scaled[i, 2]*np.sin(frac*ph))

        surf_pts_bt[i, 2] = (surf_pts_scaled[i, 0]*np.sin(frac*th) + surf_pts_scaled[i, 1]*np.sin(
            frac*ph)*np.cos(frac*th) + surf_pts_scaled[i, 2]*np.cos(frac*ph)*np.cos(frac*th))

    srep_pts_bt = np.zeros(srep_pts_scaled.shape)
    for i in range(srep_pts_scaled.shape[0]):
        if srep_pts_scaled[i, 0] > 0:
            frac = srep_pts_scaled[i, 0] / xmax
        else:
            frac = np.abs(srep_pts_scaled[i, 0]) / xmin

        srep_pts_bt[i, 0] = (
            srep_pts_scaled[i, 0]*np.cos(frac*th) - srep_pts_scaled[i, 2]*np.sin(frac*th))

        srep_pts_bt[i, 1] = (
            srep_pts_scaled[i, 1]*np.cos(frac*ph) - srep_pts_scaled[i, 2]*np.sin(frac*ph))

        srep_pts_bt[i, 2] = (srep_pts_scaled[i, 0]*np.sin(frac*th) + srep_pts_scaled[i, 1]*np.sin(
            frac*ph)*np.cos(frac*th) + srep_pts_scaled[i, 2]*np.cos(frac*ph)*np.cos(frac*th))

    return surf_pts_bt, srep_pts_bt, th, ph


def create_deformed_ellipsoid_pairs(template_dir, data_dir, count=1000):
    ellipsoid_spharm_f = os.path.join(template_dir, 'ellipsoid_SPHARM.vtk')
    srep_f = os.path.join(template_dir, 'srep_upsampled.vtp')
    surf_reader = vtk.vtkPolyDataReader()
    surf_reader.SetFileName(ellipsoid_spharm_f)
    surf_reader.Update()
    surf = surf_reader.GetOutput()

    srep_reader = vtk.vtkXMLPolyDataReader()
    srep_reader.SetFileName(srep_f)
    srep_reader.Update()
    srep = srep_reader.GetOutput()

    surf_pts = numpy_support.vtk_to_numpy(surf.GetPoints().GetData())
    srep_pts = numpy_support.vtk_to_numpy(srep.GetPoints().GetData())

    surf_pts_align, srep_pts_align = center_and_align(surf_pts, srep_pts)

    for idx in tqdm(range(1, count+1)):
        surf_pts_scaled, srep_pts_scaled, axis_scales = scale_along_axes(
            surf_pts_align, srep_pts_align)
        surf_pts_bt, srep_pts_bt, th, ph = bend_and_twist(
            surf_pts_scaled, srep_pts_scaled)
        metadata = {
            "x_scale": axis_scales[0],
            "y_scale": axis_scales[1],
            "z_scale": axis_scales[2],
            "theta": th,
            "phi": ph
        }
        with open(os.path.join(data_dir, f"meta_{idx}.json"), "w") as fp:
            json.dump(metadata, fp)

        surf.GetPoints().SetData(numpy_support.numpy_to_vtk(surf_pts_bt))
        w = vtk.vtkXMLPolyDataWriter()
        # w.SetFileName(f'surf_{xscale}_{yscale}_{zscale}_{th}_{ph}.vtp')
        surf_vtp_f = os.path.join(data_dir, f'surf_{idx}.vtp')
        w.SetFileName(surf_vtp_f)
        w.SetInputData(surf)
        w.Update()

        srep.GetPoints().SetData(numpy_support.numpy_to_vtk(srep_pts_bt))
        w = vtk.vtkXMLPolyDataWriter()
        srep_vtp_f = os.path.join(data_dir, f'srep_{idx}.vtp')
        w.SetFileName(srep_vtp_f)
        w.SetInputData(srep)
        w.Update()

        # Generate binary volumes
        w2 = vtk.vtkOBJWriter()
        w2.SetFileName('temp.obj')
        w2.SetInputData(surf)
        w2.Update()

        res = vtk.vtkResampleToImage()
        res.SetInputDataObject(surf)
        res.SetUseInputBounds(True)
        res.SetSamplingDimensions(250, 250, 250)
        res.Update()
        res_i = res.GetOutput()

        # Convert surface mesh to binary image
        mesh_reader = itk.MeshFileReader.MD3.New()
        mesh_reader.SetFileName('temp.obj')
        mesh_reader.Update()

        m2b = itk.TriangleMeshToBinaryImageFilter.MD3ID3.New()
        m2b.SetInput(mesh_reader.GetOutput())
        m2b.SetSpacing(res_i.GetSpacing())
        m2b.SetOrigin(res_i.GetOrigin())
        m2b.SetSize((250, 250, 250))
        m2b.SetInsideValue(1)  # Hence Binary! Inside/outside value is 1/0
        m2b.SetOutsideValue(0)
        m2b.Update()

        surf_nrrd_f = os.path.join(data_dir, f'surf_{idx}.nrrd')
        itk.imwrite(m2b.GetOutput(), surf_nrrd_f, compression=True)

        # Convert s-rep to binary image
        im_arr = itk.array_view_from_image(m2b.GetOutput())
        im2 = itk.image_from_array(np.zeros(im_arr.shape))
        im2.CopyInformation(m2b.GetOutput())

        loc = vtk.vtkCellLocator()
        loc.SetDataSet(res_i)
        loc.BuildLocator()

        for i in range(srep.GetNumberOfPoints()):
            pt = srep.GetPoint(i)

            ind = im2.TransformPhysicalPointToIndex(pt)
            im2.SetPixel(ind, 1)  # pixel val is 1 for the srep pixels.

        im2 = itk.binary_dilate_image_filter(im2, radius=1, foreground_value=1)
        im2 = itk.binary_morphological_closing_image_filter(
            im2, radius=1, foreground_value=1)
        srep_nrrd_f = os.path.join(data_dir, f'srep_{idx}.nrrd')
        itk.imwrite(im2, srep_nrrd_f, compression=True)


def main(args):
    datadir = args.data_dir
    if not os.path.isdir(datadir):
        os.makedirs(datadir)
    num_samples = args.count
    template_dir = "./template/"
    create_deformed_ellipsoid_pairs(template_dir, datadir, num_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str,
                        default="../data/train_3d/", help="Path to dataset dir.")
    parser.add_argument('--count', '-c', type=int,
                        default=1000, help="Number of samples to generate")
    args = parser.parse_args()
    main(args)
