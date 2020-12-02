# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:43:56 2020

@author: 12437
"""

import cv2
import os
import numpy as np 
#import shapely
#from shapely.geometry import Polygon,MultiPoint  #多边形

def switch_pixel(h, s, v): #根据像素点颜色判断所属类别
    if ((h>=0 and h<10) or (h>156 and h<=180)) and (s>=43 and s<=255) and (v>=46 and v<=255): # 红色
        return 1,h,s,v # 剪力墙类别
    elif (h>=0 and h<=180) and (s>=0 and s<43) and (v>=46 and v<=220): # 灰色
        return 2,h,s,v #普通墙类别
    elif (h>=35 and h<=77) and (s>=43 and s<=255) and (v>=46 and v<=255): # 绿色
        return 3,h,s,v #门窗类别
    elif (h>=100 and h<=124) and (s>=43 and s<=255) and (v>=46 and v<=255): # 蓝色
        return 4,h,s,v #户外门洞类别
    else:
        return 0,h,s,v #背景类别
    
def switch_image(array): #判断图像中所有像素点的类别，并将剪力墙与填充墙元素的像素坐标点分离
    array_newS = np.zeros((array.shape[0],array.shape[1],3)) #新的剪力墙的矩阵
    array_newI = np.zeros((array.shape[0],array.shape[1],3)) #新的填充墙的矩阵
    pixel_numS,pixel_numI = 0,0 #统计像素点个数
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array_newS[i][j][0],array_newS[i][j][1],array_newS[i][j][2] = 0,0,255 #白色
            array_newI[i][j][0],array_newI[i][j][1],array_newI[i][j][2] = 0,0,255 #白色
            pixeltype,h,s,v = switch_pixel(array[i][j][0], array[i][j][1], array[i][j][2])
            if pixeltype == 1: #剪力墙
                array_newS[i][j][0],array_newS[i][j][1],array_newS[i][j][2] = h,s,v
                pixel_numS += 1
            elif pixeltype == 2: #普通墙
                array_newI[i][j][0],array_newI[i][j][1],array_newI[i][j][2] = h,s,v
                pixel_numI += 1
                
    return array_newS,array_newI,pixel_numS,pixel_numI

def StoIratio(img_dir):
    img_org1 = cv2.resize(cv2.imread(img_dir),(1024,512)) #读取图像，并resize
    img_hsv1 = cv2.cvtColor(img_org1, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    img_hsv2_S,img_hsv2_I,areaS,areaI = switch_image(img_hsv1) # 分离出剪力墙和填充墙图像，以及对应面积
    StoIratio = areaS/(areaI+areaS) #剪力墙在总墙体中的面积占比
    img_hsv3_S,img_hsv3_I = np.array(img_hsv2_S,dtype='uint8'),np.array(img_hsv2_I,dtype='uint8')
    img_bgr2_S = cv2.cvtColor(img_hsv3_S, cv2.COLOR_HSV2BGR)
    img_bgr2_I = cv2.cvtColor(img_hsv3_I, cv2.COLOR_HSV2BGR)
    result_dir = img_dir+"wall\\"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    cv2.imwrite(result_dir+"img_shearwall.png",img_bgr2_S) # 输出剪力墙图
    cv2.imwrite(result_dir+"img_infillwall.png",img_bgr2_I) # 输出剪力墙图
    return StoIratio

def output_txt(StoIratios,ratios,txtpath):
    mean_ratios = np.mean(ratios)
    std_ratios = np.std(ratios)  
    txtmeanstd=open(txtpath,"w+")
    txtmeanstd.write("mean value: %f"%mean_ratios + "\n")
    txtmeanstd.write("std value: %f"%std_ratios + "\n\n")
    for StoIratio in StoIratios:
        txtmeanstd.write(str(StoIratio) + "\n")
    txtmeanstd.close()
    
## 运行函数
if __name__ == '__main__':
    imgpath = ".\\images_eval"
    
    out_path = ".\\results\\SWratios"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    imgfiles = os.listdir(imgpath)
    real_SWratios,pre_SWratios,real_ratios,pre_ratios = [], [], [], []
    if len(imgfiles) == 0 or len(imgfiles) == 1:
        print("no img files")
    else:
        for img_dir in imgfiles:
            if img_dir.split(')')[-1] == ".png":
                real_ratios.append(StoIratio(imgpath+"\\"+img_dir))
                real_ratiodict = {"img":img_dir,"ratio":real_ratios[-1]}
                real_SWratios.append(real_ratiodict)
            elif img_dir.split('_')[-1] == "image.png":
                pre_ratios.append(StoIratio(imgpath+"\\"+img_dir))
                pre_ratiodict = {"img":img_dir,"ratio":pre_ratios[-1]}
                pre_SWratios.append(pre_ratiodict)
    
    real_SWratiospath = out_path+"\\SWratios_real.txt"
    pre_SWratiospath = out_path+"\\SWratios_pre.txt"
    output_txt(real_SWratios,real_ratios,real_SWratiospath)
    output_txt(pre_SWratios,pre_ratios,pre_SWratiospath)
                