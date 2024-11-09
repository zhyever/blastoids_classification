
# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li

import itertools

import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmengine import print_log
import timm

from estimator.registry import MODELS
from estimator.models import build_model

import matplotlib.pyplot as plt
    
@MODELS.register_module()
class ClassificationModel(nn.Module):
    def __init__(
        self, 
        encoder_name,
        loss,
        num_classes=4):
        
        super().__init__()
        
        self.model = timm.create_model(encoder_name, pretrained=True)
        print_log("Model input mean {}".format(self.model.default_cfg['mean']), logger='current')
        print_log("Model input std {}".format(self.model.default_cfg['std']), logger='current')

        if 'resnet' in encoder_name:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.loss = build_model(loss)
        
        
        
    def forward(
        self,
        mode,
        image=None,
        label=None):

        if mode == 'train':
            logits = self.model(image)
            loss = self.loss(logits, label)
            loss_dict = dict()
            loss_dict['total_loss'] = loss
            
            return loss_dict, {}
            
        else:
            
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)
            
            return probs, {}