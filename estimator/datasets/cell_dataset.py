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

@DATASETS.register_module()
class CellDataset(Dataset):
    def __init__(
        self,
        mode,
        data_root,
        num_classes,
        split_file,
        input_size=512,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]):
        
        self.dataset_name = 'cell'
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
        
        self.num_classes = num_classes
        
        with open(split_file, 'r') as json_file:
            img_infos = json.load(json_file)
        
        self.file_path = []
        self.label = []
        for info in img_infos:
            if info['label'] != '[F]' and len(info['label']) == 3:
                self.file_path.append(info['img_path'])
                self.label.append(info['label'])
        
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
        label = self.label[idx]
        
        image = cv2.imread(osp.join(self.data_root, file), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        

        image = self.transform(image)
        # image = torch.from_numpy(image).float().unsqueeze(dim=0).permute(0, 3, 1, 2)
        # image = F.interpolate(image, (512, 512), mode='bilinear', align_corners=True)
        # image = image.squeeze().permute(1, 2, 0).numpy()
        # image = to_tensor(image)
        
        label = torch.tensor(self.label_map[label]).long()

        return_dict = {
            'image': image,
            'label': label,
            'filename': file}
        
        return return_dict
            
    def __len__(self):
        return len(self.file_path)
    
    def get_metrics(self, target, output, topk=(1, 2)):
        
        output = output.detach().cpu()
        target = target.detach().cpu()
        
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res['top_{}'.format(k)] = correct_k.mul_(100.0 / batch_size).item()
        
        # cm 
        cm = np.zeros((self.num_classes, self.num_classes))
        cm[target[0], pred[0]] = 1
        res['cm'] = cm

        return res
        
    
    def pre_eval_to_metrics(self, pre_eval_results):
        aggregate = []
        for item in pre_eval_results:
            aggregate.append(item.values())
        pre_eval_results = aggregate
            
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        pre_eval_results = tuple(zip(*pre_eval_results))
        ret_metrics = OrderedDict({})

        ret_metrics['top_1'] = np.nanmean(pre_eval_results[0])
        ret_metrics['top_2'] = np.nanmean(pre_eval_results[1])
        ret_metrics['cm'] = sum(pre_eval_results[2])
        

        ret_metrics = {metric: value for metric, value in ret_metrics.items()}

        return ret_metrics

    def evaluate(self, results, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        
        eval_results = {}
        # test a list of files
        ret_metrics = self.pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = (len(ret_metrics) - 1) // 2
        for i in range(num_table):
            names = ret_metric_names[i*2: i*2 + 2]
            values = ret_metric_values[i*2: i*2 + 2]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 7)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Evaluation Summary: \n' + summary_table_data.get_string(), logger='current')

        print_log('Confusion Matrix: \n {}'.format(ret_metrics['cm']), logger='current')
        
        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results

# if __name__ == '__main__':
#     d = CellDataset('train', '/ibex/ai/home/liz0l/codes/datasets/zj_project')
#     d[0]