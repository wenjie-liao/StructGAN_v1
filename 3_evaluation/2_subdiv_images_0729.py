# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:00:07 2020

@author: 12437
"""

import numpy as np 
import cv2
import os

def divide_method(img,m,n):#分割成m行n列
    h, w = img.shape[0],img.shape[1] # 读取图像shape
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m)) # 对图像进行切片操作
    gx=np.round(gx).astype(np.int) # 将切片后数据格式转化整数
    gy=np.round(gy).astype(np.int)

    divide_image = np.zeros([m-1, n-1, int(h*1.0/(m-1)+0.5), int(w*1.0/(n-1)+0.5),3], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
#    循环给每个分块赋值
    for i in range(m-1):
        for j in range(n-1):      
            divide_image[i,j,0:gy[i+1][j+1]-gy[i][j], 0:gx[i+1][j+1]-gx[i][j],:]= img[
                gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]#这样写比a[i,j,...]=要麻烦，但是可以避免网格分块的时候，有些图像块的比其他图像块大一点或者小一点的情况引起程序出错
    return divide_image

def save_subimg(divide_image,resultpath):#
    m,n=divide_image.shape[0],divide_image.shape[1]
    imgname = resultpath.split("\\")[-1] # 原图图名
    postfix = ".png" # 图的后缀名称
    for i in range(m):
        for j in range(n):
            subimg = divide_image[i,j,:] # 将分割后的子图逐步输出
            subimg_name = resultpath + "\\" + imgname + str(i) + str(j) + postfix # + postfix # 组装得到新的子图名称，ij即为分块位置
            cv2.imwrite(subimg_name,subimg)

## 运行函数
if __name__ == '__main__':
    total_NO = 24
    for image_NO in range(total_NO):
        imgpath = ".\\images_eval\\test "
        imgname_output = imgpath+"(%d)_synthesized_image.png" %(image_NO+1)
        imgname_target = imgpath+"(%d).png" %(image_NO+1)
        result_outpath = imgpath+"(%d)" %(image_NO+1)+"-outputs"
        result_tarpath = imgpath+"(%d)" %(image_NO+1)+"-targets"
        
        if os.path.exists(imgname_output):
            if not os.path.exists(result_outpath):
                os.makedirs(result_outpath)
            img = cv2.imread(imgname_output)
            img_resize = cv2.resize (img,(1024,512))
#            cv2.imwrite("shearwall (%d)-targets-RE.png"%(image_NO+1),img_resize)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            m,n = 5,9
            divide_images = divide_method(img_resize,m,n) #图像分块
            save_subimg (divide_images,result_outpath) #图像保存
        if os.path.exists(imgname_target):
            if not os.path.exists(result_tarpath):
                os.makedirs(result_tarpath)
            img = cv2.imread(imgname_target)
            img_resize = cv2.resize (img,(1024,512))
#            cv2.imwrite("shearwall (%d)-targets-RE.png"%(image_NO+1),img_resize)
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            m,n = 5,9
            divide_images = divide_method(img_resize,m,n) #图像分块
            save_subimg (divide_images,result_tarpath) #图像保存