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

# 数据增强
def image_transform(image,save_images_dir,new_fname,suffix):

    ratio = random.random()
    # # 左右翻转
    # if ratio < 0.1:
    #     Flip_horizontal_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #     Flip_horizontal_image.save(os.path.join(save_images_dir, '{}Hflip'.format(new_fname) + suffix))
    if ratio > 0.15 and ratio < 0.2:
        # 旋转
        Rotate90_image = image.transpose(Image.ROTATE_90)
        Rotate90_image.save(os.path.join(save_images_dir, '{}Rotate90'.format(new_fname) + suffix))

    if ratio > 0.95:
        # 上下翻转
        Flip_vertical_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        Flip_vertical_image.save(os.path.join(save_images_dir, '{}Vflip'.format(new_fname) + suffix))

    if ratio > 0.55 and ratio < 0.6 :
        # 仿射变换
        AFFINE_image = image.transpose(Image.AFFINE)
        AFFINE_image.save(os.path.join(save_images_dir, '{}affine'.format(new_fname) + suffix))

    # 颜色增强
    if ratio < 0.05:
        # 1.亮度增强：增强因子为0.0产生黑色图像，为1.0保持原始图像
        brightness_factor = np.random.randint(8, 16) / 10
        brightness_image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        brightness_image.save(os.path.join(save_images_dir, '{}brightness'.format(new_fname) + suffix))
    if ratio > 0.5 and ratio < 0.65:
        # 2.对比度增强
        contrast_factor = np.random.randint(8, 16) / 10
        contrast_image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        contrast_image.save(os.path.join(save_images_dir, '{}contrast'.format(new_fname) + suffix))
    # 3.色彩饱和度增强
    # if ratio > 0.97:
    #     color_factor = np.random.randint(5, 15) / 10
    #     color_image = ImageEnhance.Color(image).enhance(color_factor)
    #     color_image.save(os.path.join(save_images_dir, '{}color'.format(new_fname) + suffix))
    # if ratio > 0.4 and ratio < 0.5:
    #     # 4.锐度增强
    #     sharp_factor = np.random.randint(8, 12) / 10
    #     sharp_image = ImageEnhance.Sharpness(image).enhance(sharp_factor)
    #     sharp_image.save(os.path.join(save_images_dir, '{}sharp'.format(new_fname) + suffix))

    return

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
                if ('Normal_' in imageName) or ('Infected_' in imageName):
                    f.write(os.path.join(AUGMENTATION_IMAGE_DIR, imageName) + ' 0' + '\n')
                else:
                    f.write(os.path.join(INPUT_DATA_Normal, imageName) + ' 0' + '\n')
            else:
                print('NOt....',imageName)

    return

if __name__ == '__main__':

    INPUT_DATA_CLL = '../dataSets/dataCell/CLL/'  # 数据路径
    INPUT_DATA_Normal = '../dataSets/dataCell/Normal_and_Infected/'

    # CLL数据
    trainSet_CLL, validationSet_CLL = split(INPUT_DATA_CLL, shuffle=True, ratio=0.67)
    # Normal数据
    trainSet_Normal, validationSet_Normal = split(INPUT_DATA_Normal, shuffle=True, ratio=0.67)
    # 对trainSet_Normal做数据增强
    AUGMENTATION_IMAGE_DIR = '../dataSets/dataCell/unCLL_train_aug/'
    if not os.path.exists(AUGMENTATION_IMAGE_DIR):
        os.makedirs(AUGMENTATION_IMAGE_DIR)

    for idx, fname in enumerate(trainSet_Normal):
        image_path = os.path.join(INPUT_DATA_Normal, fname)
        image = Image.open(image_path)
        # 保存原图
        image.save(os.path.join(AUGMENTATION_IMAGE_DIR,fname))
        new_fname = fname.replace('.jpg', '_')
        suffix = '_' + str(idx) + '.jpg'
        image_transform(image, AUGMENTATION_IMAGE_DIR,new_fname,suffix)
    trainSet_Normal_AUG = os.listdir(AUGMENTATION_IMAGE_DIR)
    # 训练集和测试集
    trainSet = trainSet_CLL + trainSet_Normal_AUG
    validationSet = validationSet_CLL + validationSet_Normal

    # 训练集txt and 测试集txt
    data_list_path = '../dataSets/dataCell/'
    write_list_to_txt(trainSet, data_list_path + 'train.txt')
    write_list_to_txt(validationSet, data_list_path + 'val.txt')


