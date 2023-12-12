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
def split(data_dir, shuffle=False, ratio=0.67):
    path_list = os.listdir(data_dir) # 每个病人的文件夹
    print('path_list: ',len(path_list))
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
            for dir in dirs:
                if dir == 'M':
                    patient_bloodImg_path = os.path.join(root,dir)
                    train_bloodImg_list = os.listdir(patient_bloodImg_path)
                    for data in train_bloodImg_list:
                        trainset.append(os.path.join(root, dir, data))

    for patientDir in validation_patients:
        patient_path = os.path.join(data_dir,patientDir)
        validation_list = os.walk(patient_path)
        for root, dirs, files in validation_list:
            for dir in dirs:
                if dir == 'M':
                    patient_bloodImg_path = os.path.join(root,dir)
                    validation_bloodImg_list = os.listdir(patient_bloodImg_path)
                    for data in validation_bloodImg_list:
                        validationset.append(os.path.join(root, dir, data))

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
    INPUT_DATA_normal = '../dataSets/dataAnnotated/normal/'

    # CLL person
    train_CLL_person, validation_CLL_person = split(INPUT_DATA_CLL, shuffle=True, ratio=0.67)
    # unCLL person
    train_normal_person, validation_normal_person = split(INPUT_DATA_normal, shuffle=True, ratio=0.67)

    # 训练集和测试集
    trainSet = train_CLL_person + train_normal_person
    validationSet = validation_CLL_person + validation_normal_person

    # print('trainSet:  ',len(trainSet))
    # print('validationSet:   ',len(validationSet))

    # 训练集txt and 测试集txt
    data_list_path = '../dataSets/dataAnnotated/dataMarrow/'
    write_list_to_txt(trainSet, data_list_path + 'train_marrow.txt')
    write_list_to_txt(validationSet, data_list_path + 'val_marrow.txt')

