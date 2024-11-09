import os
import cv2
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
from mmengine.dist import get_dist_info, collect_results_cpu, collect_results_gpu
from mmengine import print_log
from estimator.utils import colorize
import torch.nn.functional as F
from tqdm import tqdm
from mmengine.utils import mkdir_or_exist
import copy

class Tester:
    """
    Tester class
    """
    def __init__(
        self, 
        config,
        runner_info,
        dataloader,
        model):
       
        self.config = config
        self.runner_info = runner_info
        
        self.dataloader = dataloader
        self.model = model
        
        self.collect_input_args = config.collect_input_args
    
    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
    
    @torch.no_grad()
    def run(self):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
                    
            batch_data_collect = self.collect_input(batch_data)
            result, log_dict = self.model(mode='infer',  **batch_data_collect)
            
            # metrics = dataset.get_metrics(depth_gt, result, disp_gt_edges=boundary.detach().cpu())
            metrics = dataset.get_metrics(batch_data_collect['label'], result) # here!
            results.extend([metrics])
            
            if self.runner_info.rank == 0:
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
                    
        # collect results from all ranks
        results = collect_results_gpu(results, len(dataset))
        if self.runner_info.rank == 0:
            ret_dict = dataset.evaluate(results)
    
    
    @torch.no_grad()
    def run_consistency(self):
        
        overlap = 270
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
                    
            batch_data_collect = self.collect_input(batch_data)
            
            
            pred_depth_crops = []
            for i in range(16): # hard code
                batch_data_collect_copy = copy.deepcopy(batch_data_collect)
                
                if 'crops_image_hr' in batch_data_collect:
                    batch_data_collect_copy['crops_image_hr'] = batch_data_collect['crops_image_hr'][0, i:i+1, :, :, :] # to 1, c, h, w
                if 'crop_depths' in batch_data_collect:
                    batch_data_collect_copy['crop_depths'] = batch_data_collect['crop_depths'][0, i:i+1, :, :, :] # to 1, c, h, w
                if 'bboxs' in batch_data_collect:
                    batch_data_collect_copy['bboxs'] = batch_data_collect['bboxs'][0, i:i+1, :] # to 1 (bs), 4
                    
                loss, log_dict = self.model(mode='train', **batch_data_collect_copy)
                pred_depth = log_dict['depth_pred']
                
                pred_depth_crop = F.interpolate(
                        pred_depth, (540, 960), mode='bilinear', align_corners=True)
                pred_depth_crops.append(pred_depth_crop.squeeze())
            
            
            pred_depth = torch.zeros((2160, 3840))
            inner_idx = 0
            pred_depth_temp = []
            consistency_error_list = []
            
            for ii, x in enumerate(dataset.h_start_list): # h
                for jj, y in enumerate(dataset.w_start_list): # w
                        
                    pred_depth[int(x + int(overlap/2)): int(x+540 - int(overlap/2)), int(y + int(overlap/2)): int(y+960 - int(overlap/2))] = \
                        pred_depth_crops[inner_idx].squeeze()[int(overlap/2):-int(overlap/2), int(overlap/2):-int(overlap/2)]

                    if ii==0 and jj==0:
                        pass
                    elif ii > 0 and jj > 0:

                        adj_crop_left = pred_depth_temp[-1]
                        common_area_1 = adj_crop_left[:, -int(overlap):]
                        common_area_2 = pred_depth_crops[inner_idx][:, :int(overlap)]
                        consistency_error_left = torch.abs(common_area_1 - common_area_2).flatten()


                        adj_crop_up = pred_depth_temp[-4]
                        common_area_1 = adj_crop_up[-int(overlap):, :]
                        common_area_2 = pred_depth_crops[inner_idx][:int(overlap), :]

                        consistency_error_up = torch.abs(common_area_1 - common_area_2).flatten()
                        consistency_error_list.append(consistency_error_left)
                        consistency_error_list.append(consistency_error_up)

                    elif ii == 0 and jj > 0: # only left

                        adj_crop = pred_depth_temp[-1]
                        common_area_1 = adj_crop[:, -int(overlap):]
                        common_area_2 = pred_depth_crops[inner_idx][:, :int(overlap)]

                        consistency_error = torch.abs(common_area_1 - common_area_2).flatten()
                        consistency_error_list.append(consistency_error)

                    
                    elif jj == 0 and ii > 0: # only up

                        adj_crop = pred_depth_temp[-4]
                        common_area_1 = adj_crop[-int(overlap):, :]
                        common_area_2 = pred_depth_crops[inner_idx][:int(overlap), :]

                        consistency_error = torch.abs(common_area_1 - common_area_2).flatten()
                        consistency_error_list.append(consistency_error)

                    pred_depth_temp.append(pred_depth_crops[inner_idx].squeeze())
                    
                    inner_idx += 1
            
            consistency_error_tensor = torch.cat(consistency_error_list)
            consistency_error = consistency_error_tensor.mean().detach().cpu().numpy()
            

            if self.runner_info.show:
                # color_pred = colorize(result, cmap='magma_r')[:, :, :3][:, :, [2, 1, 0]]
                color_pred = colorize(pred_depth)
                cv2.imwrite(os.path.join(self.runner_info.show_dir, '{}.png'.format(batch_data['img_file_basename'][0])), color_pred)
                
            # metrics = dataset.get_metrics(depth_gt, result, disp_gt_edges=boundary.detach().cpu())
            metrics = {'consistency_error': consistency_error}
            
            results.extend([metrics])
            
            if self.runner_info.rank == 0:
                batch_size = 1 * world_size
                for _ in range(batch_size):
                    prog_bar.update()
                    
        # collect results from all ranks
        results = collect_results_gpu(results, len(dataset))
        if self.runner_info.rank == 0:
            ret_dict = dataset.evaluate_consistency(results)
    
    
    @torch.no_grad()
    def run_save(self):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
                    
            batch_data_collect = self.collect_input(batch_data)
            result, log_dict = self.model(mode='infer',  **batch_data_collect)
            
            if self.runner_info.rank == 0:
                
                probs = F.softmax(result, dim=1)

                maxk = max((1, 2))
                _, pred = result.topk(maxk, 1, True, True)
                pred = pred.t()
                prediction = pred[0, 0]
                
                if prediction == 0:
                    label = 'A'
                elif prediction == 1:
                    label = 'B'
                elif prediction == 2:
                    label = 'C'
                elif prediction == 3:
                    label = 'D'
                elif prediction == 4:
                    label = 'W'
                else:
                    raise ValueError('Unknown label')
                
                with open(self.runner_info.save_name, 'a') as f:
                    f.write("{} {} {} {} {} {} {}\n".format(batch_data['filename'][0], label, probs[0, 0], probs[0, 1], probs[0, 2], probs[0, 3], probs[0, 4]))
                
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
                    
        # collect results from all ranks
        results = collect_results_gpu(results, len(dataset))
    