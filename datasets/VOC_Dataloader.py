#!/usr/bin/env python

import torch
import random
import PIL.Image
import collections
import numpy as np
import os.path as osp
from torch.utils import data
from fcn.utils import label2rgb
import matplotlib.pyplot as plt


class VOCSeg(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    class_weights = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    class_ignore = 0

    def __init__(self, root, split='train', dataset='o', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.datasets = collections.defaultdict()
        # class 0 is the background
        self.n_classes = 21

        self.datasets['o'] = osp.join(self.root, 'Original_Images')

        self.datasets['bg1'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_1')
        self.datasets['bg2'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_2')
        self.datasets['bg3'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_3')
        self.datasets['bg4'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_4')
        self.datasets['bg5'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_5')

        self.datasets['bm10'] = osp.join(self.root, 'Degraded_Images', 'Blur_Motion', 'degraded_parameter_10')
        self.datasets['bm20'] = osp.join(self.root, 'Degraded_Images', 'Blur_Motion', 'degraded_parameter_20')
        self.datasets['hi1'] = osp.join(self.root, 'Degraded_Images', 'Haze_I', 'degraded_parameter_1')
        self.datasets['ho1'] = osp.join(self.root, 'Degraded_Images', 'Haze_O', 'degraded_parameter_1')
        self.datasets['ns0.02'] = osp.join(self.root, 'Degraded_Images', 'Noise_Speckle', 'degraded_parameter_0.02')
        self.datasets['nsp0.02'] = osp.join(self.root, 'Degraded_Images', 'Noise_Salt_Pepper', 'degraded_parameter_0.02')

        img_dataset_dir = osp.join(self.root, self.datasets[dataset])

        self.files = collections.defaultdict(list)

        for split in ['train', 'val']:
            imgsets_file = osp.join(root, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(img_dataset_dir, 'VOC_train_images/%s.jpg' % did)
                lbl_file = osp.join(root, 'VOC_train_gt/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
        imgsets_file = osp.join(root, 'test.txt')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(img_dataset_dir, 'VOC_test_images/%s.jpg' % did)
            lbl_file = osp.join(root, 'VOC_test_gt/%s.png' % did)
            self.files['test'].append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        random_crop = False
        if random_crop:
            size = (np.array(lbl.shape) * 0.8).astype(np.uint32)
            img, lbl = self.random_crop(img, lbl, size)
        random_flip = False
        if random_flip:
            img, lbl = self.random_flip(img, lbl)

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        # convert to color images
        # lbl = self.label_to_color_image(lbl)
        lbl = lbl.astype(np.uint8)
        return img, lbl

    def label_to_color_image(self, lbl):
        return label2rgb(lbl)

    def random_crop(self, img, lbl, size):
        h, w = lbl.shape
        th, tw = size
        if w == tw and h == th:
            return img, lbl
        x1 = random.randint(0, w-tw)
        y1 = random.randint(0, h-th)
        img = img[y1:y1+th, x1:x1+tw, :]
        lbl = lbl[y1:y1+th, x1:x1+tw]
        return img, lbl

    def random_flip(self, img, lbl):
        if random.random() < 0.5:
            return np.flip(img, 1).copy(), np.flip(lbl, 1).copy()
        return img, lbl


# For code testing
if __name__ == "__main__":
    root = '/home/dg/Dropbox/Datasets/VOC'
    dataset = VOCSeg(root, split='test', dataset='bg2', transform=True)
    img, lbl = dataset.__getitem__(1)
    img, lbl = dataset.untransform(img, lbl)
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(lbl)
    plt.show()
