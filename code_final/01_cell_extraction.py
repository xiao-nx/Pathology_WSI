#coding="utf-8"
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import re

# 解析XML文件
def read_xml(xml_path):
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
        # 第二层节点的标签名称和属性
        #print(child.tag,":", child.attrib)
        # 遍历xml文档的第三层
        #print(child.iter("name").find("abnormal").text)
        #x = child.find("name").text
        #print("name:  ",x)

        for x in child.iter("name"):
            if x.text == 'infected':

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

# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(imagePath):
      cv_image = cv2.imdecode(np.fromfile(imagePath,dtype=np.uint8),-1)
      return cv_image

if __name__ == '__main__':
    # 数据的输入和输出路径
    data_path_dict = {
        "imageInput": "../dataSets/dataAnnotated/Infected/",
        "imageCell": "../dataSets/dataCell/Infected/"
    }
    # 批量读取文件夹数据
    imagePath = data_path_dict["imageInput"]
    imageCellPath = data_path_dict["imageCell"]

    if not os.path.exists(imageCellPath):
        os.makedirs(imageCellPath)

    # 获取每个图像的路径
    fileList = os.walk(imagePath)
    imageList = []
    for root, dirs, files in fileList:
        for file in files:
            if file.endswith('.jpg'):
                imageList.append(os.path.join(root, file))

    # 读取每张图像并操作
    for idx, imageName in enumerate(imageList):
        image = cv_imread(imageName)
        print(imageName)

        xml_path = imageName.replace('jpg','xml')
        bndboxes = read_xml(xml_path)

        patientID = (re.findall(r"20(.+?).jpg",imageName.replace("\\","_")))[0].replace("\\","-")

        for i in range(len(bndboxes)):
            xmin = int(bndboxes[i][0])
            ymin = int(bndboxes[i][1])
            xmax = int(bndboxes[i][2])
            ymax = int(bndboxes[i][3])
            cell_image = image[ymin:ymax, xmin:xmax]
            imgSaveName = imageCellPath + patientID + '_' + 'Infected_' + str(i) +'.jpg' # 保存图像的路径
            cv2.imencode('.jpg', cell_image)[1].tofile(imgSaveName)



