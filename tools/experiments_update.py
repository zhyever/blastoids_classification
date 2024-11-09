
import pandas as pd
import cv2
import os.path as osp
import random
import json

data_root='/ibex/ai/home/liz0l/codes/datasets/zj_project'
data = ['update_data/labels_my-project-name_2024-02-28-01-19-13.csv']
infos = []

for i in range(len(data)):
    d = pd.read_csv(osp.join(data_root, data[i]), header=None)
    file_path = d[0].tolist()
    label = d[1].tolist()
    
    for f, l in zip(file_path, label):
        
        update_path = osp.join('update_data', 'images', f)
        update_label = l
        
        info = dict(img_path=update_path, label=update_label)
        infos.append(info)

current_train = json.load(open('/ibex/ai/home/liz0l/codes/datasets/zj_project/train.json'))
current_train.extend(infos)

with open('/ibex/ai/home/liz0l/codes/datasets/zj_project/train_update.json', 'w') as json_file:
    json.dump(current_train, json_file, indent=4)

