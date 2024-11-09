
import pandas as pd
import cv2
import os.path as osp
import random
import json

data_root='/ibex/ai/home/liz0l/codes/datasets/zj_project'
data = ['folder_1/labels_test1_2024-01-26-12-15-12.csv', 'folder_2/labels_test2_2024-01-26-03-04-29.csv', 'folder_3/labels_test3_2024-01-26-04-58-42.csv']
infos = []

for i in range(len(data)):
    d = pd.read_csv(osp.join(data_root, data[i]), header=None)
    file_path = d[0].tolist()
    label = d[1].tolist()
    
    for f, l in zip(file_path, label):
        
        update_path = osp.join('folder_{}'.format(i+1), 'images', f)
        update_label = l
        
        info = dict(img_path=update_path, label=update_label)
        infos.append(info)

random.shuffle(infos)

train_infos = infos[:int(len(infos)*0.85)]
val_infos = infos[int(len(infos)*0.85):]

with open('/ibex/ai/home/liz0l/codes/datasets/zj_project/train.json', 'w') as json_file:
    json.dump(train_infos, json_file, indent=4)

with open('/ibex/ai/home/liz0l/codes/datasets/zj_project/val.json', 'w') as json_file:
    json.dump(val_infos, json_file, indent=4)

