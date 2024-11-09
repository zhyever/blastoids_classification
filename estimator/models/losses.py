import copy
import kornia

import torch
import torch.nn as nn
from mmengine import print_log
import torch.nn.functional as F

from estimator.registry import MODELS
from kornia.losses import dice_loss, focal_loss

@MODELS.register_module()
class ClsLoss(nn.Module):
    def __init__(self, weight):
        super(ClsLoss, self).__init__()
        self.name = 'ClsLoss'
        self.weight = torch.tensor(weight).cuda()
        
    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight)
    
    