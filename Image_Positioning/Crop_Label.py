# rotate and crop labeled images
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import shutil
import imutils
import argparse
import time
import random
import math
import glob
from PIL import Image

# 6H：最小match是4个
# 1-5H：最小match是10个
MIN_MATCH_COUNT = 10
count_num = 0
template_image = None
defect_path = "/tf/算法组/detect_images/dot_val/val_pic/"
crop_defect_path = "/tf/算法组/detect_images/dot_val/crop_val_pic/"
label_path = "/tf/算法组/detect_images/dot_val/val_lb/"
crop_label_path = "/tf/算法组/detect_images/dot_val/crop_val_lb/"

for filename in os.listdir(defect_path):
    if filename[-4:-1]==".pn":
        print("filename: %d",filename)
        template_image = cv2.imread("/tf/算法组/detect_images/Template.png", 0)
        detect_image = cv2.imread(str(defect_path+filename))
        #label_image = cv2.imread(str(os.listdir(label_path))+filename,0)
        label_image = cv2.imread(str(label_path+filename))

        count_num = count_num + 1
        # 使用SIFT检测角点
        sift = cv2.xfeatures2d.SIFT_create()

        # 获取关键点和描述符
        kp1, des1 = sift.detectAndCompute(template_image, None)
        kp2, des2 = sift.detectAndCompute(detect_image, None)

        # 定义FLANN匹配器
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 使用KNN算法匹配
        matches = flann.knnMatch(des1, des2, k=2)


        # 去除错误匹配
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        # 单应性
        if len(good) > MIN_MATCH_COUNT:
            # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
            src_pts_old = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            src_pts_list = []
            #转换坐标系为左上角坐标系
            for pnt in src_pts:
                src_pts_list.append(((pnt[0] + 900, pnt[1] + 227))) #900, 227

            src_pts = np.asarray(src_pts_list)
            src_pts = src_pts.reshape(-1, 1, 2)

            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # findHomography 函数是计算变换矩阵
            # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
            # 返回值：M 为变换矩阵，mask是掩模
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # ravel方法将数据降维处理，最后并转换成列表格式
            matchesMask = mask.ravel().tolist()

            src_pts_new_x = []
            src_pts_new_y = []
            dst_pts_new_x = []
            dst_pts_new_y = []
            count_pts = 0
            for k in range(len(src_pts)):
                if matchesMask[k]==1:
                    count_pts = count_pts + 1
                    src_pts_new_x.append(src_pts_old[k][0][0])
                    src_pts_new_y.append(src_pts_old[k][0][1])
                    dst_pts_new_x.append(dst_pts[k][0][0])
                    dst_pts_new_y.append(dst_pts[k][0][1])

            s=0
            for i in range(count_pts-2):#x1不用遍历到最后一个数字
                for j in range(i+1,count_pts-1):
                    s1=src_pts_new_x[j]-src_pts_new_x[i]
                    s2=src_pts_new_y[j]-src_pts_new_y[i]#先求出x2-x1和y2-y1，避免后面重复计算
                    for k in range(j+1,count_pts):
                        s4=abs(s1*(src_pts_new_y[k]-src_pts_new_y[i])-(src_pts_new_x[k]-src_pts_new_x[i])*s2)/2
                        if s<s4:
                            s=s4
                            idx_0 = i
                            idx_1 = j
                            idx_2 = k


            src_pts_new = np.float32([[src_pts_new_x[idx_0],src_pts_new_y[idx_0]], [src_pts_new_x[idx_1],src_pts_new_y[idx_1]], [src_pts_new_x[idx_2],src_pts_new_y[idx_2]]])
            dst_pts_new = np.float32([[dst_pts_new_x[idx_0],dst_pts_new_y[idx_0]], [dst_pts_new_x[idx_1],dst_pts_new_y[idx_1]], [dst_pts_new_x[idx_2],dst_pts_new_y[idx_2]]])

            print("src_pts_new: ", src_pts_new)
            print("dst_pts_new: ", dst_pts_new)
            M_new = cv2.getAffineTransform(dst_pts_new, src_pts_new)

            affined_image = cv2.warpAffine(detect_image, M_new, (detect_image.shape[1], detect_image.shape[0]))
            affined_image_lb = cv2.warpAffine(label_image, M_new, (detect_image.shape[1], detect_image.shape[0]))

            affined_image_bb = cv2.rectangle(affined_image, (203, 211), (1184, 432), (255,255,255), 2)
            affined_image_lb_bb = cv2.rectangle(affined_image_lb, (203, 211), (1184, 432), (255,255,255), 2)

            affined_image_crop = affined_image_bb[211:432, 203:1184]
            affined_image_lb_crop = affined_image_lb_bb[211:432, 203:1184]

            cv2.imwrite(crop_defect_path + filename+"croppic.png", affined_image_crop)
            cv2.imwrite(crop_label_path + filename+"croplb.png", affined_image_lb_crop)  

        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

print("for loop complete!!!!!!!!!!!")
