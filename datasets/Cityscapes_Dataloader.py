#!/usr/bin/env python

import torch
import random
import PIL.Image
import collections
import numpy as np
import os.path as osp
from torch.utils import data
import matplotlib.pyplot as plt

ignore_label = 255

class CityscapesSeg(data.Dataset):

    class_names = np.array([
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic_light',
        'traffic_sign',
        'vegetation',
        'terrain',
        'sky',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
        'void',
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
        0,
    ])

    class_colors = np.array([
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (0, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 0),
    ])
    # TODO: Provided by Inferno. Need to check if this is bgr or rgb: current is BGR
    # mean_bgr = np.array([0.4326707089857, 0.4251328133025, 0.41189489566336])*255
    # TODO: Provided by Inferno. Need to check if std is used.
    # std_bgr = np.array([0.28284674400252, 0.28506257482912, 0.27413549931506])*255
    # TODO: Provided by MeetShah. Maybe this is not correct.
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    class_ignore = 19

    def __init__(self, root, split='train', dataset='o', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.datasets = collections.defaultdict()
        # class 19 (the 20th class) is the ignored class
        self.n_classes = 20
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

        self.datasets['o'] = osp.join(self.root, 'Original_Images')
        self.datasets['bg1'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_1')
        self.datasets['bm1'] = osp.join(self.root, 'Degraded_Images', 'Blur_Motion', 'degraded_parameter_1')
        self.datasets['hi1'] = osp.join(self.root, 'Degraded_Images', 'Haze_I', 'degraded_parameter_1')
        self.datasets['ho1'] = osp.join(self.root, 'Degraded_Images', 'Haze_O', 'degraded_parameter_1')
        self.datasets['ns1'] = osp.join(self.root, 'Degraded_Images', 'Noise_Speckle', 'degraded_parameter_1')
        self.datasets['nsp1'] = osp.join(self.root, 'Degraded_Images', 'Noise_Salt_Pepper', 'degraded_parameter_1')

        img_dataset_dir = osp.join(self.root, self.datasets[dataset])

        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(root, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(img_dataset_dir, 'Cityscapes_train_images/%s.png' % did)
                lbl_file = osp.join(root, 'Cityscapes_train_gt/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })
        imgsets_file = osp.join(root, 'test.txt')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(img_dataset_dir, 'Cityscapes_test_images/%s.png' % did)
            lbl_file = osp.join(root, 'Cityscapes_test_gt/%s.png' % did)
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

        mask_copy = lbl.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[lbl == k] = v
        lbl = PIL.Image.fromarray(mask_copy.astype(np.uint8))
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = 19  # 20th class is ignored as 'void'

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        random_crop = False
        if random_crop:
            size = (np.array(lbl.shape)*0.8).astype(np.uint32)
            img, lbl = self.random_crop(img, lbl, size)
        random_flip = False
        if random_flip:
            img, lbl = self.random_flip(img, lbl)

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        # img /= self.std_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        # img *= self.std_bgr
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        # convert to color lbl
        # lbl = self.label_to_color_image(lbl)
        lbl = lbl.astype(np.uint8)
        return img, lbl

    def label_to_color_image(self, lbl):
        if type(lbl) is np.ndarray:
            lbl = torch.from_numpy(lbl)
        color_lbl = torch.zeros(3, lbl.size(0), lbl.size(1)).byte()
        for i, color in enumerate(self.class_colors):
            mask = lbl.eq(i)
            for j in range(3):
                color_lbl[j].masked_fill_(mask, color[j])
        color_lbl = color_lbl.numpy()
        color_lbl = np.transpose(color_lbl, (1, 2, 0))
        return color_lbl

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
    root = '/home/dg/Dropbox/Datasets/CamVid'
    dataset = CamVidSeg(root, split='train', dataset='o', transform=True)
    img, lbl = dataset.__getitem__(1)
    img, lbl = dataset.untransform(img, lbl)
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(lbl)
    plt.show()

    # dataset = CamVidSeg(root, split='train', dataset='o', transform=False)
    # mean_img = np.zeros((360, 480, 3))
    # for i in range(dataset.__len__()):
    #     img, lbl = dataset.__getitem__(i)
    #     mean_img += img
    # mean_img.transpose(2, 0, 1)
    # print (np.mean(mean_img[0]/dataset.__len__()))
    # print (np.mean(mean_img[1]/dataset.__len__()))
    # print (np.mean(mean_img[2]/dataset.__len__()))
