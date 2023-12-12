#coding="utf-8"
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import re

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
            print(image_path)
            image = cv_imread(image_path)
            print(image_path)

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
            elif ('normal' in imageName) or ('infected' in imageName):
                f.write(os.path.join(data_dir, imageName) + ' 0' + '\n')
            else:
                print('Data NOt Exist!....',imageName)

    return

if __name__ == '__main__':
    # 数据的输入和输出路径
    data_path_dict = {
        "train_data": "../dataSets/dataAnnotated/train.txt",
        "val_data": "../dataSets/dataAnnotated/val.txt",
        "train_cells": "../dataSets/dataCells/trainset/",
        "val_cells": "../dataSets/dataCells/valset/",
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

    get_cell_roi(train_list, save_path= train_cells)
    get_cell_roi(val_list,save_path= val_cells)
    save_txt_dir = '../dataSets/dataCells/'
    write_cells_path_to_txt(train_cells,save_txt_dir, 'train.txt')
    write_cells_path_to_txt(val_cells, save_txt_dir, 'val.txt')





