import random
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image

# 按比例划分文件夹下的数据
def split(data_dir, shuffle=False, ratio=0.8):

    path_list = os.walk(data_dir)
    all_list = []
    for root, dirs, files in path_list:
        for file in files:
            all_list.append(file)

    num = len(all_list)
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    if shuffle:
        random.shuffle(all_list)  # 列表随机排序
    trainset = all_list[:offset]
    validationset = all_list[offset:]

    return trainset, validationset

INPUT_DATA_CLL = '../dataSets/dataCell/CLL/'  # 数据路径
INPUT_DATA_Normal = '../dataSets/dataCell/Normal_aug/'

# CLL数据
trainSet_CLL, validationSet_CLL = split(INPUT_DATA_CLL, shuffle=True, ratio=0.67)
# Normal数据
trainSet_Normal, validationSet_Normal = split(INPUT_DATA_Normal, shuffle=True, ratio=0.67)

# 训练集和测试集
trainSet = trainSet_CLL + trainSet_Normal
validationSet = validationSet_CLL + validationSet_Normal

# 把列表写成txt
def write_list_to_txt(data_list,txt_name):
    with open(txt_name, 'w') as f:
        for imageName in data_list:
            if 'CLL' in imageName:
                f.write(os.path.join(INPUT_DATA_CLL,imageName) + ' 1' + '\n')
            if 'Normal' in imageName:
                f.write(os.path.join(INPUT_DATA_Normal,imageName) + ' 0' + '\n')

    return

data_list_path = '../dataSets/dataCell/'
write_list_to_txt(trainSet,data_list_path + 'train.txt')
write_list_to_txt(validationSet,data_list_path + 'val.txt')

