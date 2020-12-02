# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:43:56 2020

@author: 12437
"""

import cv2
import os
import numpy as np 
import shapely
from shapely.geometry import Polygon,MultiPoint  #多边形

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
def switch_image(array,ele): #判断图像中所有像素点的类别，并将不同元素的像素坐标点分离
    array_new = np.zeros((array.shape[0],array.shape[1]))
    wall_area = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
             array_new[i][j],h,s,v = switch_pixel(array[i][j][0], array[i][j][1], array[i][j][2])
             if array_new[i][j] != ele:
                 array[i][j][0], array[i][j][1], array[i][j][2] = 0,0,255
             else:
                 wall_area += 1
    return array,array_new,wall_area

def array2list(array):
#    newshape = array.shape[0]*array.shape[1]*array.shape[2]
    newarray = array.reshape((1,-1))
    newlist = newarray.tolist()
    return newlist

def contour_det(img_dir,ele_NO,result_dir): # 边缘检测
#    img_org1 = cv2.resize(cv2.imread(img_dir),(1024,512)) #resize输出与目标图像大小一致
    img_org1 = cv2.imread(img_dir)
    ex_bound = 20 # 扩从边界范围
    img_org2 = cv2.copyMakeBorder(img_org1,ex_bound,ex_bound,ex_bound,ex_bound,cv2.BORDER_CONSTANT,value=[255,255,255]) #将边界扩充20pix,保证轮廓提取质量
    img_orgH2,img_orgW2 = img_org2.shape[0],img_org2.shape[1]
    img_hsv1 = cv2.cvtColor(img_org2, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    img_hsv2,img_coord,wall_area = switch_image(img_hsv1,ele_NO) # 图像中仅保留ele_NO=n的像素
    img_brg1 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2BGR) # 将HSV转为BGR格式
    img_gray1 = cv2.cvtColor(img_brg1, cv2.COLOR_BGR2GRAY) # 将BGR格式转换为灰度图
    ret,img_gray2 = cv2.threshold(img_gray1, 127, 255, cv2.THRESH_BINARY) # 基于threshold的二值化
    contours, hierarchy = cv2.findContours(img_gray2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 边缘检测
#    cv2.imwrite(result_dir+"\\"+"img_gray2.png",img_gray2) # 输出灰度图
    return contours,img_orgH2,img_orgW2,img_org2,wall_area

def contour_wash(contours,result_dir,img_orgH,img_orgW,img_org2): # 清洗检测边缘、输出坐标
    newcontours = [] #新建列表储存后续筛选后边缘
    for i in range (len(contours)):
        #如果轮廓是四边形，且首坐标是(0,0)，则表明为外边框；或者边界坐标点小于3，排除该轮廓
        if (contours[i].shape[0]==4 and contours[i][0,:,:].all()==np.zeros((1,2)).all()) or (contours[i].shape[0]<3):
            continue
        else:
            new_list = array2list(contours[i]) # 将检测的边缘结果转化为列表
            ploy_coords=np.array(new_list).reshape(-1, 2)   # n边形二维坐标表示
            poly = Polygon(ploy_coords).convex_hull  #python 凸n边形对象，会自动计算坐标
            poly_area = poly.area #求解多边形面积
            total_area = img_orgH*img_orgW
            if poly_area/total_area < 0.0001: # 如果边缘围城的面积小于0.1%原始图像面积，去除该边缘
                continue
            else:
                newcontours.append(contours[i])
    return newcontours

def getIandU(real_cont,predict_cont2):
    inter_poly,inter_area = [],[]
    for j,pre_cont in enumerate (predict_cont2):         
        a = np.array(real_cont).reshape(-1, 2)   #多边形二维坐标表示
#        poly1 = Polygon(a).convex_hull  #python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
        poly1 = Polygon(a)
#        poly1_out = str(poly1)
        b = np.array(pre_cont).reshape(-1, 2)
#        poly2 = Polygon(b).convex_hull
        poly2 = Polygon(b)
#        poly2_out = str(poly2)
        
        # 判断是否相交
        if not poly1.intersects(poly2): #如果两四边形不相交
            inter_area.append(0)
        else:
            try:
                inter_poly.append(str(poly1.intersection(poly2))) #相交坐标
                temp_inter_area = poly1.intersection(poly2).area   #相交面积
                inter_area.append(temp_inter_area)

            except shapely.geos.TopologicalError:
                print('shapely.geos.Topological Error occured, inter_area set to 0')
                inter_area.append(0) # 相交面积取0，并集取realline的面积
                
    if np.sum(inter_area) == 0:
        inter_areas = 0
    else:
        inter_areas = np.sum(inter_area)

    return inter_areas,inter_poly

def getIoU(inter_areas,union_areas):
    if len(inter_areas) == 0:
        IoU = 0
    else:
        I = np.sum(inter_areas)
        U = np.sum(union_areas)-I
        IoU = I/U       
    return IoU

def polytocoords(poly):
    coords = []
    temp_coords1 = poly.split("((")[-1]
    temp_coords2 = temp_coords1.split("))")[0]
    temp_coords3 = temp_coords2.split(",")
    for temp_coord in temp_coords3:
        coord = temp_coord.split()
        try:
            coord1,coord2 = int(np.rint(float(coord[0]))),int(np.rint(float(coord[-1])))
            coords.append([coord1,coord2])
        except Exception as error:
            print(error)
    
    coords_array = np.array(coords).reshape(-1,1,2)
    
    return coords_array
 

## 运行函数
if __name__ == '__main__':
    total_NO = 24
    total_case = 1
    for case in range (total_case):
        IoUs = []
        for img_NO in range(total_NO):
            real_imgpath = ".\\images_eval\\test (%d)" %(img_NO+1)+"-targets"
            pre_imgpath = ".\\images_eval\\test (%d)" %(img_NO+1)+"-outputs"
            if os.path.exists(real_imgpath) and os.path.exists(pre_imgpath):
                result_dir = ".\\results\\IoU\\test(%d)"%(img_NO+1) #创建结果文件夹
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)     
                    
                full_IoU = []
                full_IoU_txtname = result_dir+"\\"+"IoU(%d).txt"%(img_NO+1)
                full_IoU_txt = open(full_IoU_txtname, "w+")
    #            full_IoU_mean_std = open(".\\IoU\\L1_7\\full_IoU_mean_std.txt", "w+")
                
                inter_areas,union_areas = [],[]
                for m in range (4):
                    for n in range (8):
                        real_img = real_imgpath+"\\test (%d)"%(img_NO+1)+"-targets"+str(m)+str(n)+".png"
                        pre_img = pre_imgpath+"\\test (%d)"%(img_NO+1)+"-outputs"+str(m)+str(n)+".png"
                        if os.path.exists(real_imgpath) and os.path.exists(pre_imgpath):
                            ele_NO = 1 #需要提取的元素编号，0-background(white),1-shearwall(red),2-wall(gray),3-win&door(green),4-gate(blue)   
                            real_cont1,Rimg_orgH2,Rimg_orgW2,Rimg_org2,wall_area_real = contour_det(real_img,ele_NO,result_dir) # 检测边缘
                            predict_cont1,Pimg_orgH2,Pimg_orgW2,Pimg_org2,wall_area_pre = contour_det(pre_img,ele_NO,result_dir) # 检测边缘
                            real_cont2 = contour_wash(real_cont1,result_dir,Rimg_orgH2,Rimg_orgW2,Rimg_org2) # 边缘清洗
                            predict_cont2 = contour_wash(predict_cont1,result_dir,Pimg_orgH2,Pimg_orgW2,Pimg_org2) # 边缘清洗
                            
                            img_org1 = cv2.imread(real_img)
                            img_orgH,img_orgW = img_org1.shape[0],img_org1.shape[1]
                            canvas = np.ones((Rimg_orgH2,Rimg_orgW2,3), dtype = "uint8")*255 # 生成空白画布
                            
                            for single_real_cont2 in real_cont2:
                                cv2.polylines(canvas, [single_real_cont2], True, (255, 0, 0)) # 真实剪力墙布置为蓝色
                            for single_predict_cont2 in predict_cont2:
                                cv2.polylines(canvas, [single_predict_cont2], True, (0, 0, 255)) # 生成剪力墙布置为红色
    
                            # 以真实图像为基础进行IoU计算
                            if len(real_cont2) != 0:
                                for i,real_cont in enumerate(real_cont2):
                                    inter_area,inter_polys = getIandU(real_cont,predict_cont2)
                                    inter_areas.append(inter_area)
                                    for inter_poly in inter_polys:
                                        inter_coords = polytocoords(inter_poly)
                                        cv2.polylines(canvas, [inter_coords], True, (0, 255, 0)) # 交集为绿色
                            cv2.imwrite((result_dir+"\\test (%d)"%(img_NO+1))+str(m)+str(n)+".png" ,canvas)
                            union_areas.append((wall_area_real+wall_area_pre))
                
                IoU = getIoU(inter_areas,union_areas)
                IoUs.append(IoU)
                
                # OUTPUT IOU
                full_IoU_txtname = result_dir+"\\"+"IoU(%d).txt"%(img_NO+1)
                full_IoU_txt = open(full_IoU_txtname, "w+")
                full_IoU_txt.write("SIoU: " + str(IoU) +"\n")
                for i,inter_area in enumerate(inter_areas):
                    full_IoU_txt.write("inter_area: "+str(inter_area)+"\n")
                for j,union_area in enumerate(union_areas):
                    full_IoU_txt.write("union_area: "+str(union_area)+"\n")
                full_IoU_txt.close()
                
    #            full_IoU_mean_std.write
            