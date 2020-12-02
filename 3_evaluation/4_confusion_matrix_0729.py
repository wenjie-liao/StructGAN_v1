# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:05:53 2020

@author: Administrator
"""

import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
from tensorflow.python.ops import math_ops, array_ops

## 定义所需函数
# objective detection function
def switch_pixel(h, s, v): #根据像素点颜色判断所属类别
    if ((h>=0 and h<10) or (h>156 and h<=180)) and (s>=43 and s<=255) and (v>=46 and v<=255): # 红色
        return 1 # 剪力墙类别
    elif (h>=0 and h<=180) and (s>=0 and s<43) and (v>=46 and v<=220): # 灰色
        return 2 #普通墙类别
    elif (h>=35 and h<=77) and (s>=43 and s<=255) and (v>=46 and v<=255): # 绿色
        return 3 #门窗类别
    elif (h>=100 and h<=124) and (s>=43 and s<=255) and (v>=46 and v<=255): # 蓝色
        return 4 #户外门洞类别
    else:
        return 0 #背景类别
    
def switch_image(array): #判断图像中所有像素点的类别
    array_new = np.zeros((array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
             array_new[i][j] = switch_pixel(array[i][j][0], array[i][j][1], array[i][j][2])
    return array_new

# Evaluation based on PA, IoU
def eval_perform(cm_array, name):
    """Compute the mean intersection-over-union via the confusion matrix."""
    # Transfer numpy to tensor
    cm_tensor = tf.convert_to_tensor(cm_array)
#    print("cm_tensor=",type(cm_tensor))
    
    # Compute using tensor
    sum_over_row = math_ops.to_float(math_ops.reduce_sum(cm_tensor, 0))
    sum_over_col = math_ops.to_float(math_ops.reduce_sum(cm_tensor, 1))
    cm_diag = math_ops.to_float(array_ops.diag_part(cm_tensor)) # 交集,对角线值
    denominator = sum_over_row + sum_over_col - cm_diag # 分母，即并集
    
    # The mean is only computed over classes that appear in the label or prediction tensor. If the denominator is 0, we need to ignore the class.
    num_valid_entries = math_ops.reduce_sum(math_ops.cast(math_ops.not_equal(denominator, 0), dtype=tf.float32)) # 类别个数
    
    ## for IoU
    # If the value of the denominator is 0, set it to 1 to avoid zero division.
    denominator = array_ops.where(math_ops.greater(denominator, 0), denominator, array_ops.ones_like(denominator))
    iou = math_ops.div(cm_diag, denominator) # 各类IoU
    
    # If the number of valid entries is 0 (no classes) we return 0.
    miou_tensor = array_ops.where(math_ops.greater(num_valid_entries, 0),math_ops.reduce_sum(iou, name=name) / num_valid_entries, 0) #mIoU
    
    # weight iou by liaowj
    weight1 = 0.4
    weight2 = 0.4
    weight3 = 0.1
    weight4 = 0.1
    weight0 = 0.0
    Wiou_tensor = weight0*iou[0] + weight1*iou[1] + weight2*iou[2] + weight3*iou[3] + weight4*iou[4]
    
    ## for PA: pixel accuracy
    PA_tensor = math_ops.div(math_ops.reduce_sum(cm_diag), math_ops.reduce_sum(sum_over_row))
    
#    创建session，执行计算
    sess = tf.Session()
    sess.run(miou_tensor)
    # tensor转化为numpy数组
#    sum_over_row_array = sum_over_row.eval(session=sess)
#    cm_diag_array = cm_diag.eval(session=sess)
    ious_array = iou.eval(session=sess)
    miou_array = miou_tensor.eval(session=sess)
    Wiou_array = Wiou_tensor.eval(session=sess)
    PA_array = PA_tensor.eval(session=sess)
    
    return ious_array, miou_array, Wiou_array, PA_array


## 运行函数
if __name__ == '__main__':
    for resultNO in range(1):
        total_NO = 24
        for img_NO in range(total_NO):
            root_imgpath1 = ".\\images_eval\\"
            real_imgpath = root_imgpath1+"test (%d).png" %(img_NO+1)
            pre_imgpath = root_imgpath1+"test (%d)_synthesized_image.png" %(img_NO+1)
            if os.path.exists(real_imgpath) and os.path.exists(pre_imgpath):
    #            result_dir = ".\\IoU\\trained\\trained(%d)"%(img_NO+1) #创建结果文件夹
                root_imgpath2 = ".\\results\\confusion_matrix\\"
                result_dir = root_imgpath2+"test(%d)"%(img_NO+1) #创建结果文件夹
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                pixIoU_txtname = result_dir+"\\"+"pixIoU(%d).txt"%(img_NO+1)
                pixIoU_txt = open(pixIoU_txtname, "w+")
    
                ## 图像数据读入与格式转化
                # 图像变为RGB矩阵
                imageR = real_imgpath
                imageP = pre_imgpath
                arrayR0 = cv.resize(cv.imread(imageR),(1024,512)) #读入图像数据,RGB,255,并统一大小
                arrayP0 = cv.resize(cv.imread(imageP),(1024,512)) #读入图像数据,RGB,255,并统一大小
                # RGB矩阵变为HSV矩阵
                arrayR1 = cv.cvtColor(arrayR0, cv.COLOR_BGR2HSV)
                arrayP1 = cv.cvtColor(arrayP0, cv.COLOR_BGR2HSV)
                
                # 根据HSV的范围判断每个像素的类别
                arrayR2 = switch_image(arrayR1) 
                arrayP2 = switch_image(arrayP1)
                
                # 将HSV矩阵reshape
                arrayR3 = np.reshape(arrayR2,(-1))
                arrayP3 = np.reshape(arrayP2,(-1))
                
                ## IoU计算部分
                num_cla = 5 # 像素的类别一共5类
                
                # confusion matrix
                conf_matrix=confusion_matrix(arrayR3, arrayP3)
                
                # computing IoU
                #mean_iou = tf.metrics.mean_iou(arrayR3, arrayP3, num_cla)
                name = 'useless'
                IoUs_test, MIoU_test, Wiou_test, PA_test = eval_perform(conf_matrix, name)
                
                pixIoU_txt.write("MIoU:" + str(MIoU_test) +"\n")
                pixIoU_txt.write("Wiou:" + str(Wiou_test) +"\n")
                pixIoU_txt.write("Pixel acc:" + str(PA_test) +"\n")
                pixIoU_txt.write("IoUs:" + "\n")
                for iou in IoUs_test:
                    pixIoU_txt.write(str(iou) + "\n")
                    
                pixIoU_txt.close()

