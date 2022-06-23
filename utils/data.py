import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms


def get_resize_transform(resize: int):
    return transforms.Compose([
        transforms.Resize(resize, resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),
                             std=(0.5, ))
    ])


def get_basic_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


class Skel2dDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, masks_dir: str, resize=None, load_in_ram=True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.resize = resize
        self.load_in_ram = load_in_ram
        self.transform = get_resize_transform(
            resize) if self.resize else get_basic_transform()
        self.ids = [splitext(file)[0] for file in listdir(images_dir)]
        self.ids.sort() # FOR consistency
        if not self.ids:
            raise RuntimeError(
                f'No image/mask files found in {images_dir} or {masks_dir}!!')
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        # Can preprocess the data and load it into RAM here.
        if self.load_in_ram:
            mask_img_pairs = [self.get_img_and_mask(
                idx) for idx in range(len(self.ids))]
            self.masks = []
            self.images = []
            for mask_f, img_f in mask_img_pairs:
                self.masks.append(self.load_img(mask_f[0]))
                self.images.append(self.load_img(img_f[0]))

    def __len__(self):
        return len(self.ids)

    def load_img(self, filename):
        return Image.open(filename).convert('L')  # convert to grayscale

    def get_img_and_mask(self, idx):
        # img_name = self.img_ids[idx]
        # mask_name = self.mask_ids[idx]
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        mask_file = list(self.masks_dir.glob(name + '.*'))
        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
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

        assert img.size == mask.size, f'Image and mask for {self.ids[idx]} should be the same size, but are {img.size} and {mask.size}'

        img = self.transform(img)
        mask = self.transform(mask)
        name = self.ids[idx]          

        return {
            'image': img,
            'mask': mask,
            'name': name
        }


def sk_loader(im_root, gt_root, batch_size=4, shuffle=True, num_worker=2, pin_memory=False, debug=False, num_debug=10):
    dataset = Skel2dDataset(im_root, gt_root)
    if debug and num_debug > 0:
        subset_ds = Subset(dataset, np.arange(num_debug))
        data_loader = DataLoader(dataset=subset_ds,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_worker,
                                pin_memory=pin_memory)
    else:
        data_loader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_worker,
                                pin_memory=pin_memory)
    return data_loader
