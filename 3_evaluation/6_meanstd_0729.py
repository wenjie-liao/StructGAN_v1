# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:06:57 2020

@author: 12437
"""
import numpy as np
import os

def read_IoUs(path):
    total_NO = 24
    IoUs = []
    for img_NO in range(total_NO):
        result_dir = path+"test(%d)"%(img_NO+1)
        if os.path.exists(result_dir): #判断结果文件夹是否存在
            IoU_path = result_dir+"\\IoU(%d).txt"%(img_NO+1)
            txttemp = open(IoU_path,"r")
            lines = txttemp.readlines()
            for i,line in enumerate(lines):
                if i==0:
                    templine = line.split(":")[-1]
                    IoUs.append(float(templine))
                    
    mean_IoUs = np.mean(IoUs)
    std_IoUs = np.std(IoUs)
    
    return IoUs,mean_IoUs,std_IoUs

def read_PAs(path):
    total_NO = 24
    MIoU,WIoU,PAs = [],[],[]
    for img_NO in range(total_NO):
        result_dir = path+"test(%d)"%(img_NO+1)
        if os.path.exists(result_dir): #判断结果文件夹是否存在
            IoU_path = result_dir+"\\pixIoU(%d).txt"%(img_NO+1)
            txttemp = open(IoU_path,"r")
            lines = txttemp.readlines()
            for i,line in enumerate(lines):
                if i==0:
                    templine = line.split(":")[-1]
                    MIoU.append(float(templine))
                elif i==1:
                    templine = line.split(":")[-1]
                    WIoU.append(float(templine))
                elif i==2:
                    templine = line.split(":")[-1]
                    PAs.append(float(templine))
                    
    mean_MIoU = np.mean(MIoU)
    std_MIoU = np.std(MIoU)
    mean_WIoU = np.mean(WIoU)
    std_WIoU = np.std(WIoU)
    mean_PAs = np.mean(PAs)
    std_PAs = np.std(PAs)
    
    return MIoU,mean_MIoU,std_MIoU,WIoU,mean_WIoU,std_WIoU,PAs,mean_PAs,std_PAs

def output_txt(IoUs,mean,std,txtpath):
    txtmeanstd=open(txtpath,"w+")
    txtmeanstd.write("mean value: %f"%mean + "\n")
    txtmeanstd.write("std value: %f"%std + "\n\n")
    for subIoU in IoUs:
        txtmeanstd.write(str(subIoU) + "\n")
    txtmeanstd.close()
    
   
## 运行函数
if __name__ == '__main__':
    IoUpath = ".\\results\\IoU\\"
    conmatrix_path = ".\\results\\confusion_matrix\\"
    path = ".\\results\\"
    SIoU,mean_SIoU,std_SIoU = read_IoUs(IoUpath)
    MIoU,mean_MIoU,std_MIoU,WIoU,mean_WIoU,std_WIoU,PAs,mean_PAs,std_PAs = read_PAs(conmatrix_path)
    SIoUpath = path+"SIoUmean_std.txt"
#    MIoUpath = path+"MIoUmean_std.txt"
    WIoUpath = path+"WIoUmean_std.txt"
    PAspath = path+"PAsmean_std.txt"
    output_txt(SIoU,mean_SIoU,std_SIoU,SIoUpath)
#    output_txt(MIoU,mean_MIoU,std_MIoU,MIoUpath)
    output_txt(WIoU,mean_WIoU,std_WIoU,WIoUpath)
    output_txt(PAs,mean_PAs,std_PAs,PAspath)
       