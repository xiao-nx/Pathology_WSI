import random
import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import os
import random

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

# 把列表写成txt
def write_list_to_txt(data_list,txt_name):

    print('len',len(data_list))

    with open(txt_name, 'w') as f:
        for imageName in data_list:
            if 'CLL' in imageName:
                f.write(os.path.join(INPUT_DATA_CLL,imageName) + ' 1' + '\n')
            elif ('Normal' in imageName) or ('Infected' in imageName):
                f.write(os.path.join(INPUT_DATA_unCLL, imageName) + ' 0' + '\n')
            else:
                print('Data NOt Exist!....',imageName)

    return

if __name__ == '__main__':

    INPUT_DATA_CLL = '../dataSets/dataCell/CLL/'  # 数据路径
    INPUT_DATA_unCLL = '../dataSets/dataCell/unCLL/'

    if not os.path.exists(INPUT_DATA_unCLL):
        os.makedirs(INPUT_DATA_unCLL)

    # CLL数据
    trainSet_CLL, validationSet_CLL = split(INPUT_DATA_CLL, shuffle=True, ratio=0.67)
    # unCLL数据
    trainSet_unCLL, validationSet_unCLL = split(INPUT_DATA_unCLL, shuffle=True, ratio=0.67)

    # 训练集和测试集
    trainSet = trainSet_CLL + trainSet_unCLL
    validationSet = validationSet_CLL + validationSet_unCLL

    # 训练集txt and 测试集txt
    data_list_path = '../dataSets/dataCell/'
    write_list_to_txt(trainSet, data_list_path + 'train.txt')
    write_list_to_txt(validationSet, data_list_path + 'val.txt')


