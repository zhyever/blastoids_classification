import torch
# import random
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
# import torch.nn as nn
import os.path as osp
from collections import OrderedDict
from prettytable import PrettyTable
from mmengine import print_log
import copy
from estimator.registry import DATASETS
from estimator.datasets.transformers import aug_color, aug_flip, to_tensor, random_crop, aug_rotate

from timm.data import create_transform
from PIL import Image

import pandas as pd
import cv2
import json
import os

@DATASETS.register_module()
class TestDataset(Dataset):
    def __init__(
        self,
        mode,
        data_root,
        input_size=512,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]):
        
        self.dataset_name = 'test'
        self.data_root = data_root
        
        self.file_path = []
        self.label = []
        
        self.label_map = {
            '[A]': 0,
            '[B]': 1,
            '[C]': 2,
            '[D]': 3,
            '[W]': 4,
        }
        
        # self.label_map = {
        #     '[A]': 0,
        #     '[B]': 1,
        #     '[C]': 1,
        #     '[D]': 2,
        #     '[W]': 3,
        #     # '[F]': 5,
        # }
        
        filenames = os.listdir(data_root)
        
        self.file_path = []
        for i in filenames:
            self.file_path.append(os.path.join(data_root, i))
        
        if mode == 'train':
            self.transform = create_transform(
                input_size=input_size, 
                is_training=True, 
                no_aug=False, 
                ratio=(1, 1), 
                hflip=0.5, 
                vflip=0.5, 
                color_jitter=0.4, 
                auto_augment=None, 
                interpolation='bilinear',
                re_prob=0.2, 
                mean=mean, 
                std=std)
        elif mode == 'val':
            self.transform = create_transform(
                input_size=input_size, 
                is_training=False, 
                no_aug=True, 
                mean=mean, 
                std=std)
        else:
            raise ValueError('mode should be train or val')   
        
            
    def __getitem__(self, idx):
        file = self.file_path[idx]
        
        image = cv2.imread(osp.join(self.data_root, file), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        
        image = self.transform(image)
        # image = torch.from_numpy(image).float().unsqueeze(dim=0).permute(0, 3, 1, 2)
        # image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=True)
        # image = image.squeeze().permute(1, 2, 0).numpy()
        # image = to_tensor(image)

        splitted_name = self.file_path[idx].split('/')
        save_name = "{}_{}_{}".format(splitted_name[-4], splitted_name[-2], splitted_name[-1])
        
        return_dict = {
            'image': image,
            'filename': save_name}
        
        return return_dict
            
    def __len__(self):
        return len(self.file_path)
    