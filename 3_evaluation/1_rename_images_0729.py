# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:48:46 2020

@author: 12437
"""

# 批量修改文件后缀

import os

def filerename(filepath,srctype,destype):
    for path,dirlist,filelist in os.walk(filepath):
        for file in filelist:

            #防止文件名中包含.
            fullist = file.split('.')
            namelist = fullist[0:-1]
            filename = ''
            for i in namelist:
                filename = filename + i + '.' 
            # print (filename)

            curndir = os.getcwd()    #获取当前路径
            # print (curndir)

            os.chdir(path)            #设置当前路径为目标目录
            newdir = os.getcwd()    #验证当前目录
            # print (newdir)

            filetype = file.split('.')[-1]    #获取目标文件格式

            if filetype == srctype:    #修改目标目录下指定后缀的文件（包含子目录）
                os.rename(file,filename+destype)

            if srctype == '*':        #修改目标目录下所有文件后缀（包含子目录）
                os.rename(file,filename+destype)

            if srctype == 'null':    #修改目标目录下所有无后缀文件（包含子目录）
                if len(fullist) == 1:
                    os.rename(file,file+'.'+destype)

            os.chdir(curndir)    #回到之前的路径

filerename('.\\images_eval','jpg','png')