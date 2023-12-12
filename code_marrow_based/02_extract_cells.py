#coding="utf-8"
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import re
from PIL import Image
from PIL import ImageEnhance
import random

# 解析XML文件
def read_xml(xml_path,patient_type='cll'):
    bndboxes = []

    if not os.path.exists(xml_path):
        print(xml_path+'not exists!')
        return bndboxes

    # 遍历文件
    tree = ET.parse(xml_path)
    # 得到根节点
    root = tree.getroot()

    # 遍历xml文档的第二层
    for child in root:

        for x in child.iter("name"):
            if x.text == patient_type:

                for sub_children in child.iter("bndbox"):
                    # 第三层节点的标签名称和属性
                    box = []
                    #print(sub_children.tag, ":", sub_children.attrib)
                    xmin = sub_children.find("xmin").text
                    ymin = sub_children.find("ymin").text
                    xmax = sub_children.find("xmax").text
                    ymax = sub_children.find("ymax").text
                    box.append(xmin)
                    box.append(ymin)
                    box.append(xmax)
                    box.append(ymax)
                    #print('box:',box)
                    bndboxes.append(box)
                #print('bndboxes:',bndboxes)

    return bndboxes

# 读取txt中的图像路径，提取细胞区域
def get_cell_roi(txt_path,save_path):

    # 读取每张图像并操作
    with open(txt_path, "r") as f:
        images_path = f.readlines()
        for image_path in images_path:
            image_path = image_path.strip('\n')
            image = cv_imread(image_path)

            xml_path = image_path.replace('jpg','xml')
            if 'cll' in image_path:
                bndboxes = read_xml(xml_path,patient_type='abnormal')
            elif 'normal' in image_path:
                bndboxes = read_xml(xml_path, patient_type='normal')
            elif 'infected' in image_path:
                bndboxes = read_xml(xml_path, patient_type='infected')
            else:
                print("can not find labeled cell !!")

            patientID = (re.findall(r"20(.+?).jpg",image_path.replace("\\","_")))[0].replace("\\","-")

            for i in range(len(bndboxes)):
                xmin = int(bndboxes[i][0])
                ymin = int(bndboxes[i][1])
                xmax = int(bndboxes[i][2])
                ymax = int(bndboxes[i][3])
                cell_image = image[ymin:ymax, xmin:xmax]
                # 判断是CLL/Normal/Infected
                imgSaveName = save_path + patientID + '_' + str(i) + '.jpg'  # 保存图像的路径
                cv2.imencode('.jpg', cell_image)[1].tofile(imgSaveName)

    return

# 对列表中的图像做扩增
def image_transform(images_list,save_image_path):
    for idx, image_path in enumerate(images_list):
        image = Image.open(image_path)
        new_fname = (image_path.replace('.jpg', '_')).split('/')[-1]
        suffix = '_' + str(idx) + '.jpg'
        ratio = random.random()

        if ratio < 0.5:
            Flip_horizontal_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            Flip_horizontal_image.save(os.path.join(save_image_path, '{}Hflip'.format(new_fname) + suffix))
            # 上下翻转
        else:
            Flip_vertical_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            Flip_vertical_image.save(os.path.join(save_image_path, '{}Vflip'.format(new_fname) + suffix))

        # 旋转
        if ratio < 0.3:
            Rotate90_image = image.transpose(Image.ROTATE_90)
            Rotate90_image.save(os.path.join(save_image_path, '{}Rotate90'.format(new_fname) + suffix))
        if (ratio > 0.3) and (ratio < 0.4):
            Rotate180_image = image.transpose(Image.ROTATE_180)
            Rotate180_image.save(os.path.join(save_image_path, '{}Rotate180'.format(new_fname) + suffix))
        if (ratio > 0.4) and (ratio < 0.6):
            Rotate270_image = image.transpose(Image.ROTATE_270)
            Rotate270_image.save(os.path.join(save_image_path, '{}Rotate270'.format(new_fname) + suffix))
        # 仿射变换
        if (ratio > 0.6) and (ratio < 0.7):
            AFFINE_image = image.transpose(Image.AFFINE)
            AFFINE_image.save(os.path.join(save_image_path, '{}affine'.format(new_fname) + suffix))

        # 颜色增强
        # 1.亮度增强：增强因子为0.0产生黑色图像，为1.0保持原始图像
        if (ratio > 0.4) and (ratio < 0.7):
            brightness_factor = np.random.randint(8, 16) / 10
            brightness_image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            brightness_image.save(os.path.join(save_image_path, '{}brightness'.format(new_fname) + suffix))
        # 2.对比度增强
        if (ratio > 0.7) and (ratio < 0.8):
            contrast_factor = np.random.randint(8, 16) / 10
            contrast_image = ImageEnhance.Contrast(image).enhance(contrast_factor)
            contrast_image.save(os.path.join(save_image_path, '{}contrast'.format(new_fname) + suffix))
        # 3.色彩饱和度增强
        if (ratio > 0.8) and (ratio < 0.9):
            color_factor = np.random.randint(5, 15) / 10
            color_image = ImageEnhance.Color(image).enhance(color_factor)
            color_image.save(os.path.join(save_image_path, '{}color'.format(new_fname) + suffix))
        # 锐度增强
        if ratio > 0.8:
            sharp_factor = np.random.randint(8, 12) / 10
            sharp_image = ImageEnhance.Sharpness(image).enhance(sharp_factor)
            sharp_image.save(os.path.join(save_image_path, '{}sharp'.format(new_fname) + '_' + str(idx) + '.jpg'))


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(imagePath):
      cv_image = cv2.imdecode(np.fromfile(imagePath,dtype=np.uint8),-1)
      return cv_image

# 把列表写成txt
def write_cells_path_to_txt(data_dir, save_dir, txt_name):

    with open(save_dir + txt_name, 'w') as f:
        data_list = os.listdir(data_dir)
        for imageName in data_list:
            if 'cll' in imageName:
                f.write(os.path.join(data_dir,imageName) + ' 1' + '\n')
            elif ('normal' in imageName):
                f.write(os.path.join(data_dir, imageName) + ' 0' + '\n')
            else:
                print('Data NOt Exist!....',imageName)

    return

if __name__ == '__main__':
    # 数据的输入和输出路径
    data_path_dict = {
        "train_data": "../dataSets/dataAnnotated/dataMarrow/train_marrow.txt",
        "val_data": "../dataSets/dataAnnotated/dataMarrow/val_marrow.txt",
        "train_cells": "../dataSets/dataMarrowCells/trainset/",
        "val_cells": "../dataSets/dataMarrowCells/valset/",
    }
    # 批量读取文件夹数据
    train_list = data_path_dict["train_data"]
    val_list = data_path_dict["val_data"]

    train_cells = data_path_dict['train_cells']
    val_cells = data_path_dict['val_cells']

    if not os.path.exists(train_cells):
        os.makedirs(train_cells)
    if not os.path.exists(val_cells):
        os.makedirs(val_cells)

    # 从原图中扣取细胞区域
    get_cell_roi(train_list, save_path= train_cells)
    get_cell_roi(val_list,save_path= val_cells)

    # 保存训练数据和验证数据的路径
    save_txt_dir = '../dataSets/dataMarrowCells/'
    write_cells_path_to_txt(train_cells,save_txt_dir, 'train_raw.txt')
    write_cells_path_to_txt(val_cells, save_txt_dir, 'val.txt')

    # 对训练集的normal组细胞进行数据增强
    normal_images_list = []
    cll_images_list = []
    train_cells_txt = '../dataSets/dataMarrowCells/train_raw.txt'
    with open(train_cells_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()[:-2]
            if 'normal' in line:
                normal_images_list.append(line)
            elif 'cll' in line:
                cll_images_list.append(line)
            else:
                print("wrong!~~~")
    print('normal_images_list: ',len(normal_images_list))
    print('cll_images_list:  ',len(cll_images_list))
    image_transform(normal_images_list, train_cells)

    # # 重新写训练集的txt
    write_cells_path_to_txt(train_cells,save_txt_dir, 'train.txt')
