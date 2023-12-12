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
import torch.nn.functional as F

from xlwt import *


def write_excel(dict_data):

    file = Workbook(encoding='utf-8') # 指定file以utf-8的格式打开
    table = file.add_sheet('data') # 指定打开的文件名

    list_data = []

    for x in dict_data:
        # for循环将data字典中的键和值分批的保存在ldata中
        t = [x]
        for a in dict_data[x]:
            t.append(a)
        list_data.append(t)

    for i, p in enumerate(list_data):
        # 将数据写入文件,i是enumerate()函数返回的序号数
        for j, q in enumerate(p):
            # print i,j,q
            table.write(i, j, q)
    file.save('data_Normal_train.xls')

    return


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net = net.to(device)

modelpath = './checkpoints/best_model.pth'
checkpoint = torch.load(modelpath,map_location=lambda storage,loc: storage)
# for k,v in checkpoint.items():
#     print(k)
net.load_state_dict(checkpoint)
net.eval()

#net.load_state_dict(modelpath)

imagespath = '../dataset/learningData/train/Normal'
imageList = os.listdir(imagespath)

info = {}

for imageName in imageList:
    image_path = os.path.join(imagespath, imageName)

    image = Image.open(image_path)
    imgblob = data_transforms(image).unsqueeze(0)

    with torch.no_grad():
        inputs = imgblob.to(device)
        outputs = net(inputs)

        # 类别的预测概率
        softmax_layer = nn.Softmax(dim=1)
        softmax_outputs = softmax_layer(outputs)

        probability, preds = torch.max(softmax_outputs, 1)

        probability = probability.cpu().numpy()
        preds = preds.cpu().numpy()
        #print(imageName,probability,preds)
        info[imageName] = ['trainSet','1',float(probability),int(preds)]

# print(info)
write_excel(info)