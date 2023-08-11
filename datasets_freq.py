import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import ToTensor
from turbojpeg import TurboJPEG
from jpeg2dct.numpy import load, loads


def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    edge_path = os.path.join(root, 'edge')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png'), os.path.join(edge_path, img_name + '.png')) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None, edge=False, freq=True):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.edge = edge
        self.freq = freq
        self.totensor = ToTensor()
        self.kernel = np.ones((2, 2), np.uint8)
        self.jpeg = TurboJPEG()

    def __getitem__(self, index):
        img_path, gt_path, edge_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        ycbcr_image = Image.open(img_path).convert("YCbCr")
        ycbcr_image = ImageOps.exif_transpose(ycbcr_image)

        target = Image.open(gt_path).convert('L')
        target = ImageOps.exif_transpose(target)
        if self.edge:
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            edge = cv2.dilate(edge, self.kernel)
            edge = Image.fromarray(edge)

            if self.joint_transform is not None:
                img, ycbcr_image, target, edge = self.joint_transform(img, ycbcr_image, target, edge, image_path=img_path)
            if self.freq:
                ycbcr_image = self.target_transform(ycbcr_image)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
                edge = self.target_transform(edge)
            if self.freq:
                return img, ycbcr_image, target, edge, img_path
            else:
                return img, target, edge, img_path
        else:
            if self.joint_transform is not None:
                img, ycbcr_image, target = self.joint_transform(img,  ycbcr_image, target, image_path=img_path)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.freq:
                ycbcr_image = self.totensor(ycbcr_image)
                return img, ycbcr_image, target, img_path
            else:
                return img, target, img_path

    def __len__(self):
        return len(self.imgs)
