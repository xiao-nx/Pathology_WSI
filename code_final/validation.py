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

from xlwt import *


def write_excel(dict_data):

    worksheet = Workbook(encoding='utf-8') # 指定file以utf-8的格式打开
    sheet = worksheet.add_sheet('val') # 指定打开的文件名

    # 第一行
    row0 = ["imageName", "set_type", "y_true", "y_predicted","y_positive_score"]
    # 写第一行
    for i in range(0, len(row0)):
        sheet.write(0, i, row0[i])

    list_data = []

    num = [a for a in dict_data]

    for x in num:
        # for循环将data字典中的键和值分批的保存在ldata中
        t = [x]
        for a in dict_data[x]:
            t.append(a)
        list_data.append(t)

    for i, p in enumerate(list_data):
        # 将数据写入文件,i是enumerate()函数返回的序号数
        for j, q in enumerate(p):
            # print i,j,q
            sheet.write(i + 1, j, q)
    worksheet.save('results/Reset50_add_train.xls')

    return


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_transforms =  transforms.Compose([
               transforms.Resize(224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

net = models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features

# net = models.vgg16_bn(pretrained=False)
# # num_ftrs = net.classifier[6].in_features
# # feature_model = list(net.classifier.children())
# # feature_model.pop()
# # feature_model.append(nn.Linear(num_ftrs, 2))
# # net.classifier = nn.Sequential(*feature_model)

net.fc = nn.Linear(num_ftrs, 2)
net = net.to(device)

modelpath = './models/model_25_2/ResNet50_best_model.pth'
checkpoint = torch.load(modelpath,map_location=lambda storage,loc: storage)

net.load_state_dict(checkpoint)
net.eval()

imagespath = '../dataSets/dataCell/train.txt'

lines = []
images_list = open(imagespath, 'r')  # 设置文件对象
images = images_list.readlines()  # 直接将文件中按行读到list里

info = {}

for line in images:
    #lines.append(line)
    image_path = line[:-2]

    image = Image.open(image_path)
    imgblob = data_transforms(image).unsqueeze(0)

    with torch.no_grad():
        inputs = imgblob.to(device)
        outputs = net(inputs)

        softmax_layer = nn.Softmax(dim=1)
        softmax_outputs = softmax_layer(outputs)
        probability, preds = torch.max(softmax_outputs, 1)

        probability = probability.cpu().numpy()
        preds = preds.cpu().numpy()

        y_true = 1 if 'CLL' in image_path else 0
        if preds == 0:
            probability = 1- probability

        info[image_path] = ['train', y_true, int(preds), float(probability)]

write_excel(info)

images_list.close()  # 关闭文件

