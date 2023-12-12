# -*- encoding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import re
from skimage import segmentation,morphology

K = 4

'''
cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)
参数：
data: 分类数据，最好是np.float32，每个特征放一列
K: 分类数目
bestLabels：预设的分类标签或者None
criteria: 迭代停止的模式选择，是格式为（type, max_iter, epsilon）的元组
      其中type有如下模式：
      —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止
      —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止
      —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束
attempts：重复试验kmeans算法次数，将会返回最好的一次结果
flags：初始中心选择，有两种方法
       ——v2.KMEANS_PP_CENTERS
       ——cv2.KMEANS_RANDOM_CENTERS
返回值：
      compactness：紧密度，返回每个点到相应重心的距离的平方和
      labels：结果标记，每个成员被标记为0,1等
      centers：由聚类的中心组成的数组
'''

# 聚类分割
def seg_keams_color(image):
      imgH = image.shape[0]
      imgW = image.shape[1]
      imageFlat = image.reshape(imgH * imgW, 3)
      imageFlat = np.float32(imageFlat)

      # 迭代参数
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
      flags = cv2.KMEANS_RANDOM_CENTERS

      # 聚类 K
      compactness, labels, centers = cv2.kmeans(imageFlat, K, None, criteria, 10, flags)

      # 显示结果
      imageRes = labels.reshape((imgH, imgW))

      return imageRes

def showResult(image, Kmeans_label):
      # 复制一张图像
      imageCopy = image.copy()
      idx1 = np.where(Kmeans_label[:, :] == 0)
      idx2 = np.where(Kmeans_label[:, :] == 1)
      idx3 = np.where(Kmeans_label[:, :] == 2)
      idx4 = np.where(Kmeans_label[:, :] == 3)

      imageCopy[idx1] = 0
      imageCopy[idx2] = 50
      imageCopy[idx3] = 100
      imageCopy[idx4] = 150

      return imageCopy

# 筛选类别
def showLabel(image, Kmeans_label):

      # BGR图像中G分量黑色最多
      imageB, imageG, imageR = cv2.split(image)

      dst_idx = np.where(imageG[:, :] < 120)

      # 新建一张画布
      imageSegmentation = 255 * np.ones_like(imageG)

      idx_1 = np.where(Kmeans_label[:, :] == 0)
      idx_2 = np.where(Kmeans_label[:, :] == 1)
      idx_3 = np.where(Kmeans_label[:, :] == 2)
      idx_4 = np.where(Kmeans_label[:, :] == 3)

      dst_idx_num = len(dst_idx[0])

      if dst_idx_num == 0:
          return

      idx = [len(idx_1[0]),len(idx_2[0]),len(idx_3[0]),len(idx_4[0])]
      n_times = [(x /dst_idx_num)  for x in idx]
      #print('n_times:   ',n_times)
      min_times = min(n_times)
      for i in range(len(idx)):
            #print('倍数：   ',idx[i] / dst_idx_num)
            if idx[i] / dst_idx_num < math.ceil(min_times): # 像素数量最接近的
                  other_idx = np.where(Kmeans_label[:, :] != i)
                  imageSegmentation[other_idx] = 0

      return imageSegmentation

# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(imagePath):
      cv_image = cv2.imdecode(np.fromfile(imagePath,dtype=np.uint8),-1)
      return cv_image

# 高斯去噪
def Gaussian_Blur(imageGay):
    # 高斯去噪
    blurred = cv2.GaussianBlur(imageGay, (27, 27), 0)
    return blurred

def viewImage(image,windowName):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 形态学处理
def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # 执行图像形态学
    morphologyImage = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphologyImage = cv2.erode(thresh, None, iterations=4)
    morphologyImage = cv2.dilate(morphologyImage, None, iterations=4)

    return morphologyImage

def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255)

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv

	return im_out

# 目标识别
def findcnts_and_box_point(imageBinary):

    # 寻找轮廓
    morphologyImage,contours, hierarchy = cv2.findContours(imageBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 按长度删除重复的轮廓
    # delete_list = []
    # c, row, col = hierarchy.shape
    # for i in range(row):
    #     if hierarchy[0, i, 0] > 0:
    #         delete_list.append(i)
    # contours = delet_contours(contours, delete_list)

    return contours

def drawcnts_and_cut(original_img, box):
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    y_mid = y1 + math.ceil((y2 - y1) / 2)
    x_mid = x1 + math.ceil((x2 - x1) / 2)

    crop_img = original_img[y_mid - 25:y_mid + 25, x_mid - 25:x_mid + 25]

    return draw_img, crop_img

#  自定义函数：用于删除列表指定序号的轮廓
#  输入 1：contours：原始轮廓
#  输入 2：delete_list：待删除轮廓序号列表
#  返回值：contours：筛选后轮廓
def delet_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


if __name__ == '__main__':

      # 数据的输入和输出路径
      data_path_dict = {
            "imageInput": "../dataset/dataRaw/Normal",
            "imageSeg": "../dataset/dataSeg/Normal",
            "imageROI": "../dataset/dataROI/Normal",
            "imageCell": "../dataset/dataCell/Normal"
      }

      # 批量读取文件夹数据
      imagePath = data_path_dict["imageInput"]
      imageSegPagth = data_path_dict["imageSeg"]
      imageROIPath = data_path_dict["imageROI"]
      imageCellPath = data_path_dict["imageCell"]

      # 创建文件夹
      if not os.path.exists(imageSegPagth):
            os.makedirs(imageSegPagth)
      if not os.path.exists(imageROIPath):
            os.makedirs(imageROIPath)
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
      for idx,imageName in enumerate(imageList):
            image = cv_imread(imageName)
            print(imageName)

            # 聚类
            imageRes = seg_keams_color(image)
            imgSeg = showResult(image, imageRes)

            imgShow = showLabel(image, imageRes) # 二值图像

            # 保存处理后的图像
            pattern = re.compile(r'(dataRaw)(.*)(-)')
            imgSegmentationSaveName = pattern.sub(imageSegPagth + '/' + r'Normal-segmetation-', imageName) # 保存图像的路径
            cv2.imwrite(imgSegmentationSaveName,imgShow)

            # 拷贝图像
            imageBinary = imgShow.copy()

            # 高斯滤波
            imgBlur = Gaussian_Blur(imageBinary)

            #imgMor = image_morphology(imgBlur)

            # 孔洞填充
            imgFilling = fillHole(imgBlur)
            #viewImage(imgFilling, 'imgFilling')

            # 清除与边界相邻的目标物
            imgDst = segmentation.clear_border(imgFilling)
            #viewImage(imgDst, 'imgDst')

            # 目标检测
            contours = findcnts_and_box_point(imgDst)

            imageCopy = image.copy()
            cell_cnt = 0
            for c in contours:
                  if cv2.contourArea(c) > 20000:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(imageCopy,(x,y),(x+w, y+h),(0,0,255),2)

                        # 保存每个细胞
                        cell_cnt += 1
                        x1 = x
                        x2 = x1 + w
                        y1 = y
                        y2 = y1 + h
                        cell_image = image[y1:y2,x1:x2,:]

                        imgCellSaveName = pattern.sub(imageCellPath + '/' + r'Normal-cell-'+str(cell_cnt)+'-', imageName)  # 保存图像的路径
                        cv2.imwrite(imgCellSaveName, cell_image)

            # 保存处理后的图像
            imgROISaveName = pattern.sub(imageROIPath + '/' + r'Normal-ROI-', imageName) # 保存图像的路径
            cv2.imwrite(imgROISaveName,imageCopy)
