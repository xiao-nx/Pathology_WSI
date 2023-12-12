#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
#import time
import os
from PIL import Image
import numpy as np
import pandas as pd

import re


from xlwt import *


def write_to_excel(arr_1,arr_2,sava_path):

    writer = pd.ExcelWriter(sava_path)
    df1 = pd.DataFrame(arr_1)
    df2 = pd.DataFrame(arr_2)
    df1.to_excel(writer,sheet_name='train',header=False,index=False)
    df2.to_excel(writer,sheet_name='val',header=False,index=False)
    writer.close()

    return


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_transforms =  transforms.Compose([
               transforms.Resize(224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

net = models.densenet121(pretrained=True)  # 加载预训练网络
num_ftrs = net.classifier.in_features

# net = models.resnet50(pretrained=True)
# num_ftrs = net.fc.in_features

net.fc = nn.Linear(num_ftrs, 2)
net = net.to(device)

modelpath = './models/model_25/DensenNet121_best_model.pth'
checkpoint = torch.load(modelpath,map_location=lambda storage,loc: storage)

net.load_state_dict(checkpoint)
net.eval()

txts_path = '../dataSets/dataCell/data_aug/data_aug_25/'
txts = os.listdir(txts_path)

train_index = [["imageName", "set_type", "y_true", "y_predicted", "y_positive_score"]]
eval_index = [["imageName", "set_type", "y_true", "y_predicted", "y_positive_score"]]

for txt in txts:
    txt_path = os.path.join(txts_path,txt)
    print(txt_path)
    lines = []
    images_list = open(txt_path, 'r')  # 设置文件对象
    images = images_list.readlines()  # 直接将文件中按行读到list里


    imageName = []
    y_true = []
    y_predicted = []
    y_positive_score = []

    for line in images:
        # lines.append(line)
        image_path = line[:-2]
        image = Image.open(image_path)
        imgName = re.split('/|\\\\', image_path)[-1]
        #imgName = image_path.split("/")[-1]
        imgblob = data_transforms(image).unsqueeze(0)

        with torch.no_grad():
            inputs = imgblob.to(device)
            outputs = net(inputs)

            softmax_layer = nn.Softmax(dim=1)
            softmax_outputs = softmax_layer(outputs)
            probability, preds = torch.max(softmax_outputs, 1)

            probability = probability.cpu().numpy()
            y_predicted = preds.cpu().numpy()

            #print(imgName)
            y_true = 1 if 'CLL' in imgName else 0
            if y_predicted == 0:
                probability = 1 - probability

            if 'train' in txt_path:
                train_index.append([image_path, "trainset", y_true, int(y_predicted), float(probability)])
            if 'val' in txt_path:
                eval_index.append([image_path, "valset", y_true, int(y_predicted), float(probability)])

    sava_path = './models_evalution/CLL_and_unCLL/DensenNet121_add_25.xlsx'
    write_to_excel(train_index, eval_index, sava_path)

    images_list.close()  # 关闭文件




