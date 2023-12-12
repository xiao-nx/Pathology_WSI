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
    path_list = os.listdir(data_dir) # 每个病人的文件夹
    num = int(len(path_list)) # 病人文件夹的总数

    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], path_list
    if shuffle:
        random.shuffle(path_list)  # 列表随机排序
    train_patients = path_list[:offset]
    validation_patients = path_list[offset:]

    trainset = []
    validationset = []

    for patientDir in train_patients:
        patient_path = os.path.join(data_dir,patientDir)
        train_list = os.walk(patient_path)
        for root, dirs, files in train_list:
            for file in files:
                trainset.append(os.path.join(root,file))

    for patientDir in validation_patients:
        patient_path = os.path.join(data_dir,patientDir)
        validation_list = os.walk(patient_path)
        for root, dirs, files in validation_list:
            for file in files:
                validationset.append(os.path.join(root,file))

    # print(len(trainset),trainset)
    # print(len(validationset),validationset)

    return trainset, validationset


# 把列表写成txt
def write_list_to_txt(data_list, txt_name):

    with open(txt_name, 'w') as f:
        for imageName in data_list:
            if imageName.endswith('.jpg'):
                f.write(imageName + '\n')

    return


if __name__ == '__main__':

    INPUT_DATA_CLL = '../dataSets/dataAnnotated/cll/'  # 数据路径
    INPUT_DATA_unCLL = '../dataSets/dataAnnotated/normal_and_infected/'

    if not os.path.exists(INPUT_DATA_unCLL):
        os.makedirs(INPUT_DATA_unCLL)

    # CLL person
    train_CLL_person, validation_CLL_person = split(INPUT_DATA_CLL, shuffle=True, ratio=0.67)
    # unCLL person
    train_unCLL_person, validation_unCLL_person = split(INPUT_DATA_unCLL, shuffle=True, ratio=0.67)


    # 训练集和测试集
    trainSet = train_CLL_person + train_unCLL_person
    validationSet = validation_CLL_person + validation_unCLL_person

    # print(len(trainSet),trainSet)
    # print(len(validationSet),validationSet)

    # 训练集txt and 测试集txt
    data_list_path = '../dataSets/dataAnnotated/'
    write_list_to_txt(trainSet, data_list_path + 'train.txt')
    write_list_to_txt(validationSet, data_list_path + 'val.txt')

