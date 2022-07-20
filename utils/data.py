from typing import List
import logging
import os
import glob
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms
from monai.data import Dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Resized,
    RandZoomd,
    RandFlipd,
    RandRotate90d,
)
from monai.utils import set_determinism

set_determinism(seed=0)

def get_resize_transform(resize: int):
    return transforms.Compose(
        [
            transforms.Resize(resize, resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )


def get_basic_transform():
    return transforms.Compose([transforms.ToTensor(),])


class Skel2dDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, masks_dir: str, resize=None, load_in_ram=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.resize = resize
        self.load_in_ram = load_in_ram
        self.transform = (
            get_resize_transform(resize) if self.resize else get_basic_transform()
        )
        self.ids = [splitext(file)[0] for file in listdir(images_dir)]
        self.ids.sort()  # FOR consistency
        if not self.ids:
            raise RuntimeError(
                f"No image/mask files found in {images_dir} or {masks_dir}!!"
            )
        logging.info(f"Creating dataset with {len(self.ids)} examples")

        # Can preprocess the data and load it into RAM here.
        if self.load_in_ram:
            mask_img_pairs = [
                self.get_img_and_mask(idx) for idx in range(len(self.ids))
            ]
            self.masks = []
            self.images = []
            for mask_f, img_f in mask_img_pairs:
                self.masks.append(self.load_img(mask_f[0]))
                self.images.append(self.load_img(img_f[0]))

    def __len__(self):
        return len(self.ids)

    def load_img(self, filename):
        return Image.open(filename).convert("L")  # convert to grayscale

    def get_img_and_mask(self, idx):
        # img_name = self.img_ids[idx]
        # mask_name = self.mask_ids[idx]
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + ".*"))
        mask_file = list(self.masks_dir.glob(name + ".*"))
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        # print(mask_file, img_file)
        return mask_file, img_file

    def __getitem__(self, idx):
        if self.load_in_ram:
            mask = self.masks[idx]
            img = self.images[idx]
        else:
            mask_file, img_file = self.get_img_and_mask(idx)
            mask = self.load_img(mask_file[0]) / 255.0
            img = self.load_img(img_file[0]) / 255.0

        assert (
            img.size == mask.size
        ), f"Image and mask for {self.ids[idx]} should be the same size, but are {img.size} and {mask.size}"

        img = self.transform(img)
        mask = self.transform(mask)
        name = self.ids[idx]

        return {"image": img, "mask": mask, "name": name}


def sk_loader(
    im_root,
    gt_root,
    batch_size=4,
    shuffle=True,
    num_worker=2,
    pin_memory=False,
    debug=False,
    num_debug=10,
):
    dataset = Skel2dDataset(im_root, gt_root)
    if debug and num_debug > 0:
        subset_ds = torch.utils.data.Subset(dataset, np.arange(num_debug))
        data_loader = torch.utils.data.DataLoader(
            dataset=subset_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker,
            pin_memory=pin_memory,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker,
            pin_memory=pin_memory,
        )
    return data_loader


def get_surf_srep_split(
    data_dir: str, validation_frac=0.1, test_frac=0.1, random_shuffle=False, debug=False
) -> List[dict]:
    images = sorted(glob.glob(os.path.join(data_dir, "surf_*.nrrd")))
    labels = sorted(glob.glob(os.path.join(data_dir, "srep_*.nrrd")))
    if (len(images) != len(labels)) or len(images) == 0 or len(labels) == 0:
        raise AssertionError(
            "Check the data directory for equal number of data and label files!"
        )
    data_dicts = [
        {"image": image_name, "label": label_name, "fname": get_fname(image_name)}
        for image_name, label_name in zip(images, labels)
    ]

    if debug:
        data_dicts = data_dicts[:10]
        random_shuffle = False

    test_size = int(test_frac * len(data_dicts))
    validation_size = int(validation_frac * len(data_dicts))
    # indices = np.arange(len(data_dicts))
    # if random_shuffle:
    #     np.random.shuffle(indices)
    indices = list(range(len(data_dicts)))
    test_idxs = indices[:test_size]
    validation_idxs = indices[test_size : test_size + validation_size]
    train_idxs = indices[test_size + validation_size :]
    # print(test_size, validation_size)
    # print(test_idxs, validation_idxs, train_idxs)
    return (
        data_dicts[test_size + validation_size :],
        data_dicts[test_size : test_size + validation_size],
        data_dicts[:test_size],
    )


def get_srep_data_transform(resize_shape=(224, 224, 224)):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=resize_shape),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_aug_transform(resize_shape=(224, 224, 224), num_crop=4):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            # RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.6, max_zoom=1.2),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=resize_shape,
                pos=1,
                neg=1,
                num_samples=num_crop,
                image_key="image",
                image_threshold=1e-3,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.30,),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.30,),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.30,),
            RandRotate90d(keys=["image", "label"], prob=0.30, max_k=3,),
            Resized(keys=["image", "label"], spatial_size=resize_shape),
        ]
    )


def get_fname(file_path):
    fname_with_ext = os.path.basename(file_path)
    return fname_with_ext.split(".")[0]


def get_hippocampi_files(data_dir):
    images = sorted(glob.glob(os.path.join(data_dir, "*_hippo.gipl.gz")))
    data_dict = [
        {"image": image_name, "fname": get_fname(image_name)} for image_name in images
    ]
    return data_dict


def get_hippocampi_transform(resize_shape=(224, 224, 224)):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Resized(keys=["image"], spatial_size=resize_shape),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True
            ),
            ToTensord(keys=["image"]),
        ]
    )

