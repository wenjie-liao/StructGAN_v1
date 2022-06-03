# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from skimage import morphology
import shapely
from shapely.geometry import MultiPolygon, Polygon, LineString
from shapely.ops import unary_union


class Read_files():
    def __init__(self, proj_name, roots):
        self.proj_name = proj_name  # 项目名称
        self.archi_dir = os.path.join(roots[1], "archi.png")  # structgan输入图像的路径
        self.engineer_shearwall_dir = os.path.join(roots[1], "struct.png")  # 工程师设计剪力墙图像的路径
        self.gan_shearwall_img_dir = os.path.join(roots[2], "layout.png")  # structgan生成图像的路径
        self.plot_Wall_coords_dir = os.path.join(roots[0], "plot_Wall_coords.txt")  # 绘图隔墙坐标路径
        self.plot_Win_coords_dir = os.path.join(roots[0], "plot_Win_coords.txt")  # 绘图门窗坐标路径
        self.plot_Gate_coords_dir = os.path.join(roots[0], "plot_Gate_coords.txt")  # 绘图防火门坐标路径
    
    def read_vectors(self):
        self.plot_Wall_coords = list(np.loadtxt(self.plot_Wall_coords_dir))  # 绘图隔墙坐标
        self.plot_Win_coords = list(np.loadtxt(self.plot_Win_coords_dir))  # 绘图门窗坐标
        self.plot_Gate_coords = list(np.loadtxt(self.plot_Gate_coords_dir))  # 绘图防火门坐标

        return None
    
    def read_ganimg(self):
        self.archi_img = cv2.imread(self.archi_dir)  # 输入建筑图
        self.img_H,self.img_W = self.archi_img.shape[0],self.archi_img.shape[1] # 确定建筑图的尺寸
        
        self.engineer_shearwall_img = cv2.imread(self.engineer_shearwall_dir)  # 输入工程师设计结构图
        self.engineer_shearwall_img = cv2.resize(self.engineer_shearwall_img,(self.img_W,self.img_H))
        
        self.gan_shearwall_img = cv2.imread(self.gan_shearwall_img_dir)  # structgan生成剪力墙图像
        self.gan_shearwall_img = cv2.resize(self.gan_shearwall_img,(self.img_W,self.img_H))
        
        return None

    
class Structure_extract():
    def __init__(self, proj_name, roots, archi_img, gan_shearwall_img,
                 plot_Wall_coords, plot_Win_coords, plot_Gate_coords):  # 绘图外墙坐标
        self.proj_name = proj_name  # 项目名称
        self.process_image_root = roots[3]  # 图像处理过程中文件夹名称
        self.archi_img = archi_img  # OpenCV格式的建筑图像
        self.gan_shearwall_img = gan_shearwall_img  # OpenCV格式的StructGAN图像
        self.plot_Wall_coords = plot_Wall_coords  # 建筑墙坐标
        
        # 所有的建筑线汇总
        self.plot_Archi_coords = plot_Wall_coords + plot_Win_coords + plot_Gate_coords
        self.plot_Win_coords = plot_Win_coords
        self.plot_Gate_coords = plot_Gate_coords
        
        # 创建节点、墙体数组
        self.Nodes = []  # 数组中存储为节点坐标数组 {id,x,y,archi_id,is_end}
        self.Nodes_coords = []  # 数组中存储为节点坐标数组
        self.Shearwall_xs = []  # 数组中存储为x向剪力墙dictionary {id,x1,y1,x2,y2,dir,node1,node2}
        self.Shearwall_ys = []  # 数组中存储为y向剪力墙dictionary {id,x1,y1,x2,y2,dir,node1,node2}
        self.Beam_xs, self.Beam_ys = [], []  # 数组中存储为x\y向梁dictionary {id,x1,y1,x2,y2,dir,node1,node2}
        self.Shearwalls = []  # 输出数据
        self.Beam_s = []  #  输出数据

        # 设定shear wall的HSV颜色阈值
        self.hsv_lower_red1, self.hsv_lower_red2 = np.array([0, 50, 50]), np.array([160, 50, 50])
        self.hsv_upper_red1, self.hsv_upper_red2 = np.array([10, 255, 255]), np.array([180, 255, 255])
        # 设定beam的HSV颜色阈值
        # self.hsv_lower_yellow, self.hsv_upper_yellow = np.array([26,43,46]),np.array([35,255,255])
        # self.hsv_lower_green,self.hsv_upper_green = np.array([35,43,46]),np.array([77,255,255])
        # 进行墙体mask腐蚀时的卷积核尺寸
        self.ero_kernel = 2 
        
        # 设定墙线清洗参数
        self.min_struct_len = 6  # 结构构件的最短长度，unit:pix
        self.min_gap = 4  # 两个同一墙线上的墙体是否归并的最大gap，unit:pix
        self.node_gap = 4  # 提取的剪力墙端点与初始墙线端点是否归并的最大gap，unit:pix
        self.cont_pixels = 3  # 用于判断是否为剪力墙的连续像素点的个数
        self.wash_times = 1  # 清洗剪力墙次数
        self.center_range = 8  # 与中心的偏移值，unit:pix
        self.out_range = 15  # 与最近点的最大允许间隔，unit:pix
        self.grid_gap = 4  # 网格间隔，unit:pix
        self.threshold = 6  # 阈值，在阈值之内的都视为相等，但是之后修改为完全相等
    
    def seperate_shearwall(self):  # 对图像中的剪力墙元素进行剥离
        # 对图像去除噪音
        blur_img = cv2.bilateralFilter(self.gan_shearwall_img, 9, 75, 75)
        # 转换到HSV
        hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
        # 根据阈值构建掩模
        mask1 = cv2.inRange(hsv_img, self.hsv_lower_red1, self.hsv_upper_red1)
        mask2 = cv2.inRange(hsv_img, self.hsv_lower_red2, self.hsv_upper_red2)
        mask = mask1 + mask2
        # 对原图像和掩模进行位运算
        res_img = cv2.bitwise_and(blur_img, blur_img, mask=mask)
        gary_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
        ret, bin_img = cv2.threshold(gary_img, 50, 255, 0)
        # 先腐蚀再膨胀去除噪音
        kernel = np.ones((self.ero_kernel, self.ero_kernel), np.uint8)
        erobin_img = cv2.erode(bin_img, kernel, iterations=2)
        dilbin_img = cv2.dilate(erobin_img, kernel, iterations=3)
        # 存储处理后的binary图像
        # cv2.imshow('binary_img',dilbin_img)
        # cv2.waitKey(0)
        self.binimg_dir = os.path.join(self.process_image_root, self.proj_name+"_shearwall_bin.png")
        dilbin_img_reverse = cv2.bitwise_not(src=dilbin_img)
        # cv2.imwrite(self.binimg_dir, dilbin_img_reverse)
        # cv2.imwrite(self.binimg_dir, dilbin_img)
        self.dilbin_img = dilbin_img  # 二值化的剪力墙图像
        
        return None
    
    def inter_structure(self, struct_coords, bin_img, archi_id):  # 建筑墙线与剪力墙像素做交集，提取剪力墙墙线
        temp_struct_archis = []  # 在一个建筑墙线中提取的剪力墙集合
        struct_dir = ""
        x1, y1, x2, y2 = int(struct_coords[0]), int(struct_coords[1]), int(struct_coords[2]), int(struct_coords[3])
        
        if y1 == y2:  # 为x向构件，从左至右求交点提取
            struct_dir = "x"
            x_min, x_max = min(x1, x2), max(x1, x2)
            start_x, end_x = 0, 0  # 初始化起点与重点坐标值
            for inter_pt in range(x_min,x_max+1):  # 通过判断建筑墙线与墙体的交线情况，来提取剪力墙坐标
                if bin_img[y1, inter_pt] == 255 and start_x == 0:
                    start_x = inter_pt
                if bin_img[y1, inter_pt] == 0 and start_x != 0 or \
                    (bin_img[y1,inter_pt] == 255 and inter_pt == x_max):
                    end_x = inter_pt
                    struct_len = (end_x-start_x)  # 当前交线长度
                    if struct_len >= self.min_struct_len:  # 交线长度大于最小墙体长度，认为是墙体
                        # 进行节点归并
                        if abs(start_x-x_min) <= self.node_gap:  # 如果该节点与最小节点范围小于node_gap，节点归并
                            start_x = x_min
                        if abs(end_x-x_max) <= self.node_gap:  # 如果该节点与最小节点范围小于node_gap，节点归并
                            end_x = x_max
                        node1, node2 = [start_x, y1], [end_x, y1]
                        
                        # 首先判断该节点是否曾经记录
                        if node1 not in self.Nodes_coords:  # 若该节点不在节点组中
                            node1_id = len(self.Nodes_coords)
                            self.Nodes_coords.append(node1)
                            self.Nodes.append({"id": node1_id, "x": node1[0], "y": node1[1], "archi_id": archi_id})
                        else:  # 若该节点在节点组中
                            for i, temp_node in enumerate(self.Nodes_coords):
                                if node1 == temp_node:
                                    node1_id = i
                        # 首先判断该节点是否曾经记录
                        if node2 not in self.Nodes_coords:  # 若该节点不在节点组中
                            node2_id = len(self.Nodes_coords)
                            self.Nodes_coords.append(node2)
                            self.Nodes.append({"id": node2_id, "x": node2[0], "y": node2[1], "archi_id": archi_id})
                        else:  # 若该节点在节点组中
                            for i, temp_node in enumerate(self.Nodes_coords):
                                if node2 == temp_node:
                                    node2_id = i
                        # 记录该提取的剪力墙
                        # bin_img = cv2.line(bin_img,(start_x,y1),(end_x,y1),3)
                        # cv2.imshow("temp_bin",bin_img)
                        # cv2.waitKey(0)
                        temp_struct = {"x1": node1[0], "y1": node1[1], "x2": node2[0], "y2": node2[1],
                                       "dir": "x", "node1_id": node1_id, "node2_id": node2_id}
                        temp_struct_archis.append(temp_struct)
                        start_x, end_x = 0, 0  # 初始化起点与重点坐标值
        
        elif x1 == x2:  # 为y向构件，从下至上求交点提取
            struct_dir = "y"
            y_min, y_max = min(y1, y2), max(y1, y2)
            start_y, end_y = 0, 0  # 初始化起点与重点坐标值
            for inter_pt in range(y_min, y_max+1):  # 通过判断建筑墙线与墙体的交线情况，来提取剪力墙坐标
                if bin_img[inter_pt, x1] == 255 and start_y == 0:
                    start_y = inter_pt
                if (bin_img[inter_pt, x1] == 0 and start_y != 0) or \
                    (bin_img[inter_pt, x1] == 255 and inter_pt == y_max):
                    end_y = inter_pt
                    struct_len = (end_y-start_y)  # 当前交线长度
                    if struct_len >= self.min_struct_len:  # 交线长度大于最小墙体长度，认为是墙体
                        # 进行节点归并
                        if abs(start_y-y_min) <= self.node_gap:  # 如果该节点与最小节点范围小于node_gap，节点归并
                            start_y = y_min
                        if abs(end_y-y_max) <= self.node_gap:  # 如果该节点与最小节点范围小于node_gap，节点归并
                            end_y = y_max
                        node1, node2 = [x1, start_y], [x1, end_y]
                        
                        # 首先判断该节点是否曾经记录
                        if node1 not in self.Nodes_coords:  # 若该节点不在节点组中
                            node1_id = len(self.Nodes_coords)
                            self.Nodes_coords.append(node1)
                            self.Nodes.append({"id": node1_id, "x": node1[0], "y": node1[1], "archi_id": archi_id})
                        else: # 若该节点在节点组中
                            for i,temp_node in enumerate(self.Nodes_coords):
                                if node1 == temp_node:
                                   node1_id = i
                        # 首先判断该节点是否曾经记录
                        if node2 not in self.Nodes_coords:  # 若该节点不在节点组中
                            node2_id = len(self.Nodes_coords)
                            self.Nodes_coords.append(node2)
                            self.Nodes.append({"id": node2_id, "x": node2[0], "y": node2[1], "archi_id": archi_id})
                        else:  # 若该节点在节点组中
                            for i, temp_node in enumerate(self.Nodes_coords):
                                if node2 == temp_node:
                                   node2_id = i
                        # 记录该提取的剪力墙
                        # bin_img = cv2.line(bin_img,(x1,start_y),(x1,end_y),3)
                        # cv2.imshow("temp_bin",bin_img)
                        # cv2.waitKey(0)
                        temp_struct = {"x1": node1[0], "y1": node1[1], "x2": node2[0], "y2": node2[1],
                                       "dir": "y", "node1_id": node1_id, "node2_id": node2_id}
                        temp_struct_archis.append(temp_struct)
                        start_y, end_y = 0, 0  # 初始化起点与重点坐标值

        else:
            print("Warning: dialog structure not in consideration")

        return temp_struct_archis, struct_dir

    def wash_structure_topy(self):  # 根据拓扑关系清洗提取的结构构件
        # 节点归并，将后面节点与前面归并
        for i, node_i in enumerate(self.Nodes):
            for j in range((i+1), len(self.Nodes)):
                node_j = self.Nodes[j]
                if node_i["archi_id"] == node_j["archi_id"]:
                    # print("Node: %s and Node: %s"%(node_i["id"],node_j["id"]))
                    if node_i["x"] == node_j["x"]:
                        node_gap = abs(node_i["y"]-node_j["y"])
                        if node_gap <= self.node_gap:  # j向i归并
                            new_node_j_coords = [node_i["x"], node_i["y"]]
                            new_node_j = {"id": j, "x": node_i["x"], "y": node_i["y"], "archi_id": node_i["archi_id"]}
                            self.Nodes_coords[j] = new_node_j_coords
                            self.Nodes[j] = new_node_j
                            # print("Node: %s is merged"%node_j)
        
        # 剪力墙结构构件坐标调整
        # {"x1":node1[0],"y1":node1[1],"x2":node2[0],"y2":node2[1],"dir":"x","node1_id":node1_id,"node2_id":node2_id}
        for i, Shearwall_x in enumerate(self.Shearwall_xs):
            node1_id, node2_id = Shearwall_x["node1_id"], Shearwall_x["node2_id"]
            self.Shearwall_xs[i]["x1"], self.Shearwall_xs[i]["y1"] = self.Nodes_coords[node1_id][0], self.Nodes_coords[node1_id][1]
            self.Shearwall_xs[i]["x2"], self.Shearwall_xs[i]["y2"] = self.Nodes_coords[node2_id][0], self.Nodes_coords[node2_id][1]
        for i, Shearwall_y in enumerate(self.Shearwall_ys):
            node1_id, node2_id = Shearwall_y["node1_id"], Shearwall_y["node2_id"]
            self.Shearwall_ys[i]["x1"], self.Shearwall_ys[i]["y1"] = self.Nodes_coords[node1_id][0], self.Nodes_coords[node1_id][1]
            self.Shearwall_ys[i]["x2"], self.Shearwall_ys[i]["y2"] = self.Nodes_coords[node2_id][0], self.Nodes_coords[node2_id][1]
        
        return None
    
    def vect_structures(self):  # 该版本的剪力墙矢量化提取，采用先验的墙线与剪力墙二值图的交集
        # Step1 墙体与剪力墙二值图做交线
        archi_id = -1
        for i, Wall_coords in enumerate(self.plot_Wall_coords):
            archi_id += 1
            temp_shearwalls, wall_dir = self.inter_structure(Wall_coords, self.dilbin_img, archi_id)  # 提取剪力墙
            wash_shearwalls = temp_shearwalls
            if wall_dir == "x":
                self.Shearwall_xs = self.Shearwall_xs + wash_shearwalls
            elif wall_dir == "y":
                self.Shearwall_ys = self.Shearwall_ys + wash_shearwalls
            else:
                print("error!")

        # Step2 对构件进行基于拓扑关系的清洗
        self.wash_structure_topy()

        # 汇总剪力墙结果
        self.Shearwalls = [self.Shearwall_xs, self.Shearwall_ys]
        
        return None
    
#### 建筑外轮廓提取
class Archi_extract():
    def __init__(self, proj_name, roots, archi_img, archi_img_dir):  # 绘图外墙坐标
        self.proj_name = proj_name
        self.process_image_root = roots[3]  # 图像处理过程中文件夹名称
        self.archi_img = archi_img
        archi_img_name = archi_img_dir.split("\\")[-1]
        self.archi_img_dir = os.path.join(self.process_image_root, archi_img_name)
        self.archioutline_threshold = 0.1
        self.img_W,self.img_H = self.archi_img.shape[0],self.archi_img.shape[1] # 确定建筑图的尺寸
        
        return None
        
    def vect_floor(self): # 建筑外轮廓提取主函数
        #### 输入建筑图像读取-清洗的前处理
        # 图像二值化
        gary_img = cv2.cvtColor(self.archi_img,cv2.COLOR_BGR2GRAY)
        ret,bin_img = cv2.threshold(gary_img, 220, 255, cv2.THRESH_BINARY_INV)
        # mask图像膨胀，让可能存在缝隙的外轮廓全部填满
        kernel = np.ones((3,3),np.uint8)
        dilbin_img = cv2.dilate(bin_img,kernel,iterations = 3)
        # 保存bin图像
        self.binimginput_save_dir = self.archi_img_dir.split(".png")[0]+"_bin.png"
        cv2.imwrite(self.binimginput_save_dir,dilbin_img)
        # 轮廓提取
        contours, hierarchy = cv2.findContours(dilbin_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #### 计算外轮廓属性
        outline_cnt_pix,outline_cenpts_pix,cnt_props,max_area = [],[],[],0
        for i,contour in enumerate(contours):
            cnt_M = cv2.moments(contour) # 轮廓求矩
            cnt_area = cnt_M['m00'] # 轮廓面积，也可用cv2.contourArea()计算
            x_cm = int(cnt_M['m10']/cnt_M['m00']) # 轮廓中心x
            y_cm = int(cnt_M['m01']/cnt_M['m00']) # 轮廓中心y
            cnt_props.append([contour,cnt_M,cnt_area,(x_cm,y_cm)]) # 储存矩\面积\中心点元组
            if max_area < cnt_area:
                max_area = cnt_area
            else:
                continue
            if max_area >= (self.archioutline_threshold*self.img_W*self.img_H) and len(contour)>4: #如果该轮廓的面积大于整张图像面积*阈值，并且非4节点
                outline_cnt_pix = contour # 整体的外轮廓
                outline_cenpts_pix = (x_cm,y_cm) # 整体外轮廓的中心
                story_pixarea = max_area # 楼层面积
        #### 对建筑外轮廓提取结果进行判断
        # epsilon = 0.005*cv2.arcLength(outline_cnt_pix,True)
        # outline_cnt_pix = cv2.approxPolyDP(outline_cnt_pix,epsilon,True)
        outline_cnt_img = cv2.drawContours(self.archi_img,[outline_cnt_pix],-1,(255,0,0),2)
        outline_cnt_img = cv2.circle(outline_cnt_img, outline_cenpts_pix, 5, (255,0,0),-1)
        
        # 保存图像
        # cv2.imshow("outline_cnt_img",outline_cnt_img)
        # cv2.waitKey(0)
        self.imginput_save_dir = self.archi_img_dir.split(".png")[0]+"_outline.png"
        cv2.imwrite(self.imginput_save_dir,outline_cnt_img)

        # 保存外轮廓提取结果
        self.outline_cnt_pix,self.outline_cenpts_pix,self.story_pixarea = outline_cnt_pix,outline_cenpts_pix,story_pixarea
        
        return None

class Eval_byIoU():
    def __init__(self,Eng_img,GAN_structs,Eng_structs,result_root,proj_name):
        self.pix_rect_thick = 3
        self.Eng_img = Eng_img
        self.GAN_structs = GAN_structs
        self.Eng_structs = Eng_structs
        self.result_root = result_root
        self.proj_name = proj_name
        
        return None
        
    def create_rect(self, lines): # 基于构件轴线坐标创建矩形，用于后续交并比计算
        contours,contour_areas = [],[]
        for i,line in enumerate(lines):
            x1,y1,x2,y2 = line["x1"],line["y1"],line["x2"],line["y2"]
            if line["dir"]=="y": # y向构件
                pt1 = [(x1-self.pix_rect_thick),y1]
                pt2 = [(x1+self.pix_rect_thick),y1]
                pt3 = [(x2+self.pix_rect_thick),y2]
                pt4 = [(x2-self.pix_rect_thick),y2]
                line_len = abs(y2-y1)
            elif line["dir"]=="x": # x向构件
                pt1 = [x1,(y1-self.pix_rect_thick)]
                pt2 = [x1,(y1+self.pix_rect_thick)]
                pt3 = [x2,(y2+self.pix_rect_thick)]
                pt4 = [x2,(y2-self.pix_rect_thick)]
                line_len = abs(x2-x1)
            else:
                continue
            contour = [pt1,pt2,pt3,pt4]
            contours.append(contour)
            contour_area = line_len*self.pix_rect_thick*2
            contour_areas.append(contour_area)

        return contours,contour_areas

    def polyintersect(self,real_conts,predict_conts):
        inter_polys,inter_area = [],[]
        for j,pre_cont in enumerate (predict_conts):
            a = np.array(real_conts).reshape(-1, 2)   #多边形二维坐标表示
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
                    inter_poly = poly1.intersection(poly2) # 交集计算
                    temp_inter_area = inter_poly.area   #相交面积
                    inter_area.append(temp_inter_area)
                    inter_polys.append(inter_poly)
    
                except shapely.geos.TopologicalError:
                    # print('shapely.geos.Topological Error occured, inter_area set to 0')
                    inter_area.append(0) # 相交面积取0，并集取realline的面积
                    
        if np.sum(inter_area) == 0:
            inter_areas = 0
        else:
            inter_areas = np.sum(inter_area)
    
        return inter_areas,inter_polys
    
    def polytocoords(self,poly):
        ploygons = []
        if poly.geom_type == "Polygon":
            poly_wkt = poly.wkt
            poly_wkts = poly_wkt.split("POLYGON")[-1]

        elif poly.geom_type == "MultiPolygon":
            poly_wkt = poly.wkt
            poly_wkts = poly_wkt.split("MULTIPOLYGON")[-1]
        else:
            poly_wkts = ""
            
        temp_coords1 = poly_wkts.split("(")
        for i,temp_coord1 in enumerate(temp_coords1):
            if len(temp_coord1)>5:
                temp_coords2 = temp_coord1.split(")")[0]
                temp_coords3 = temp_coords2.split(",")
                coords = []
                for temp_coord in temp_coords3:
                    coord = temp_coord.split()
                    coord1,coord2 = int(np.rint(float(coord[0]))),int(np.rint(float(coord[-1])))
                    coords.append((coord1,coord2))
                coords_array = np.array(coords).reshape((-1,1,2))
                ploygons.append(coords_array)

        return ploygons
    
    def getIoU(self,inter_areas,union_areas):
        if len(inter_areas) == 0:
            IoU = 0
        else:
            I = np.sum(inter_areas)
            U = np.sum(union_areas)-I
            IoU = I/U       
        return IoU
    
    def iou_eval(self):
        inter_areas,union_areas = [],[]
        
        # 获取StructGAN设计的剪力墙构件轴线对应的矩形
        GAN_contours,GAN_contour_areas = self.create_rect(self.GAN_structs)
        # 获取工程师设计的剪力墙构件轴线对应的矩形
        Eng_contours,Eng_contour_areas = self.create_rect(self.Eng_structs)
        
        # 创建空白画布用以画图
        img_orgH,img_orgW = self.Eng_img.shape[0],self.Eng_img.shape[1]
        canvas = np.ones((img_orgH,img_orgW,3), dtype = "uint8")*255 # 生成空白画布
        
        cv2.polylines(canvas, np.array(Eng_contours), True, (255, 0, 0)) # 真实剪力墙布置为蓝色
        # cv2.imshow("Eng_contours",canvas)
        cv2.polylines(canvas, np.array(GAN_contours), True, (0, 0, 255)) # 生成剪力墙布置为红色
        # cv2.imshow("GAN_contours",canvas)
        # cv2.waitKey(0)
        
        # 以真实图像为基础获取交并集轮廓
        if len(Eng_contours) != 0:
            for i,Eng_contour in enumerate(Eng_contours):
                inter_area,inter_polys = self.polyintersect(Eng_contour,GAN_contours)
                if len(inter_polys) != 0:
                    inter_areas.append(inter_area)
                    for inter_poly in inter_polys:
                        try:
                            inter_coords = self.polytocoords(inter_poly) #交集坐标
                            cv2.polylines(canvas, inter_coords, True, (0, 255, 0)) # 交集为绿色
                        except:
                            continue
        union_areas.append((GAN_contour_areas + Eng_contour_areas))
        
        # 进行IoU计算
        if len(inter_areas) == 0:
            Shearwall_IoU = 0
        else:
            Shearwall_Inter = np.sum(inter_areas)
            Shearwall_Union = np.sum(union_areas)-Shearwall_Inter
            Shearwall_IoU = Shearwall_Inter/Shearwall_Union
        
        # 输出交并集的图像
        # cv2.imshow("IoU_contours",canvas)
        # cv2.waitKey(0)
        img_name = "%s_IoU_contours.png"%self.proj_name
        img_dir = os.path.join(self.result_root, img_name)
        cv2.imwrite(img_dir,canvas)
        
        # 输出交并集计算结果
        txt_name = "%s_IoU.txt"%self.proj_name
        txt_dir = os.path.join(self.result_root, txt_name)
        with open(txt_dir,"w") as iou_txt:
            iou_txt.write("%.8f\n"%Shearwall_IoU)
            iou_txt.write("inter_areas: %.8f\n"%Shearwall_Inter)
            iou_txt.write("union_areas: %.8f\n"%Shearwall_Union)
        
        return None
        
class Eval_byfloor():
    def __init__(self,archi_img,scale,GAN_structs,outline_cnt_pix,result_root,proj_name):
        self.archi_img = archi_img
        self.GAN_structs = GAN_structs
        self.outline_cnt_pix = outline_cnt_pix
        self.result_root = result_root
        self.proj_name = proj_name
        self.free_cantilever = 2000 # unit:mm
        self.free_cantilever_pix = self.free_cantilever/scale
        
        return None
    
    def get_cantilever(self):
        cantilever_cnts = []
        for i,GAN_struct in enumerate(self.GAN_structs):
            if GAN_struct["dir"] == "x":
                x1,y1,x2,y2 = GAN_struct["x1"],GAN_struct["y1"],GAN_struct["x2"],GAN_struct["y2"]
                x_max,x_min = max(x1,x2), min(x1,x2)
                
                # 求x_max处的弧形坐标
                # Define the arc (presumably ezdxf uses a similar convention)
                centerx_max, centery_max = x_max, y1
                radius = self.free_cantilever_pix
                start_angle, end_angle = -90, 90 # In degrees
                numsegments = 10
                # The coordinates of the arc
                theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
                x_max_arc = centerx_max + radius * np.cos(theta)
                y_max_arc = centery_max - radius * np.sin(theta)
                
                # 求x_min处的弧形坐标
                # Define the arc (presumably ezdxf uses a similar convention)
                centerx_min, centery_min = x_min, y1
                radius = self.free_cantilever_pix
                start_angle, end_angle = 90, 270 # In degrees
                numsegments = 10
                # The coordinates of the arc
                theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
                x_min_arc = centerx_min + radius * np.cos(theta)
                y_min_arc = centery_min - radius * np.sin(theta)
                
                # 行列合并
                x_arc,y_arc = np.concatenate((x_max_arc,x_min_arc)),np.concatenate((y_max_arc,y_min_arc))
                cantilever_cnt = np.column_stack([x_arc,y_arc])
                
            elif GAN_struct["dir"] == "y":
                x1,y1,x2,y2 = GAN_struct["x1"],GAN_struct["y1"],GAN_struct["x2"],GAN_struct["y2"]
                y_max,y_min = max(y1,y2), min(y1,y2)
                
                # 求y_max处的弧形坐标
                # Define the arc (presumably ezdxf uses a similar convention)
                centerx_max, centery_max = x1, y_max
                radius = self.free_cantilever_pix
                start_angle, end_angle = -180, 0 # In degrees
                numsegments = 10
                # The coordinates of the arc
                theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
                x_max_arc = centerx_max + radius * np.cos(theta)
                y_max_arc = centery_max - radius * np.sin(theta)
                
                # 求x_min处的弧形坐标
                # Define the arc (presumably ezdxf uses a similar convention)
                centerx_min, centery_min = x1, y_min
                radius = self.free_cantilever_pix
                start_angle, end_angle = 0, 180 # In degrees
                numsegments = 10
                # The coordinates of the arc
                theta = np.radians(np.linspace(start_angle, end_angle, numsegments))
                x_min_arc = centerx_min + radius * np.cos(theta)
                y_min_arc = centery_min - radius * np.sin(theta)
                
                # 行列合并
                x_arc,y_arc = np.concatenate((x_max_arc,x_min_arc)),np.concatenate((y_max_arc,y_min_arc))
                cantilever_cnt = np.column_stack([x_arc,y_arc])
            else:
                cantilever_cnt = np.array([[]])
            
            # 对坐标取整
            cnts = []
            for pt in cantilever_cnt:
                pt_x,pt_y = int(pt[0]),int(pt[1])
                cnts.append([[pt_x,pt_y]])
            cantilever_cnts.append(np.array(cnts))
                
        self.cantilever_cnts = cantilever_cnts
        
        return cantilever_cnts
    
    def polytocoords(self,poly):
        ploygons = []
        if poly.geom_type == "Polygon":
            poly_wkt = poly.wkt
            poly_wkts = poly_wkt.split("POLYGON")[-1]

        elif poly.geom_type == "MultiPolygon":
            poly_wkt = poly.wkt
            poly_wkts = poly_wkt.split("MULTIPOLYGON")[-1]
        else:
            poly_wkts = ""
            
        temp_coords1 = poly_wkts.split("(")
        for i,temp_coord1 in enumerate(temp_coords1):
            if len(temp_coord1)>5:
                temp_coords2 = temp_coord1.split(")")[0]
                temp_coords3 = temp_coords2.split(",")
                coords = []
                for temp_coord in temp_coords3:
                    coord = temp_coord.split()
                    coord1,coord2 = int(np.rint(float(coord[0]))),int(np.rint(float(coord[-1])))
                    coords.append((coord1,coord2))
                coords_array = np.array(coords).reshape((-1,1,2))
                ploygons.append(coords_array)

        return ploygons
    
    def arraytopoly(self,array):
        array_re = array.reshape((-1,2))
        cnts = []
        for pt in array_re:
            pt_x,pt_y = int(pt[0]),int(pt[1])
            cnts.append((pt_x,pt_y))
        ploy = Polygon(cnts)
        
        return ploy
        
    def floor_eval(self):
        # 创建空白画布用以画图
        img_H,img_W = self.archi_img.shape[0],self.archi_img.shape[1]
        canvas = np.ones((img_H,img_W,3), dtype = "uint8")*255 # 生成空白画布
        
        # 提取所有墙体的悬臂面积轮廓
        cantilever_cnts = self.get_cantilever()
        
        # 从floor轮廓中逐个减去悬臂面积
        polygon_outline = self.arraytopoly(self.outline_cnt_pix)
        polygon_res = self.arraytopoly(self.outline_cnt_pix) 
        for i,cantilever_cnt in enumerate(cantilever_cnts):
            polygon_cantilever = self.arraytopoly(cantilever_cnt)
            if polygon_res.intersects(polygon_cantilever): # 如果两个多边形相交
                polygon_res = polygon_res - polygon_cantilever
            else: # 两个多边形不相交
                continue
        try:
            polygon_res_cnts = self.polytocoords(polygon_res)
        except:
            polygon_res_cnts = np.array([[]]).reshape((-1,1,2))
        
        # 计算残差面积与指标        
        polygon_res_area = polygon_res.area # 残差面积
        polygon_outline_area = polygon_outline.area # 楼板面积
        res_ratio = polygon_res_area/polygon_outline_area
        
        # 绘制悬臂面积
        canvas = np.ones((img_H,img_W,3), dtype = "uint8")*255 # 生成空白画布
        cv2.drawContours(canvas,[np.array(self.outline_cnt_pix)],-1,(255,0,0),2)# 楼板轮廓绘制为蓝色
        cv2.drawContours(canvas,cantilever_cnts,-1,(0,0,255),1) # 悬臂板绘制为红色
        cv2.drawContours(canvas,polygon_res_cnts,-1,(0,255,0),3) # 二者相减的残差绘制为绿色
        
        # 输出图像
        img_name = "%s_cantilever_contours.png"%self.proj_name
        img_dir = os.path.join(self.result_root, img_name)
        cv2.imwrite(img_dir,canvas)
        
        # 输出交并集计算结果
        txt_name = "%s_cantilever.txt"%self.proj_name
        txt_dir = os.path.join(self.result_root, txt_name)
        with open(txt_dir,"w") as iou_txt:
            iou_txt.write("%.8f\n"%res_ratio)
            iou_txt.write("res_areas: %.8f\n"%polygon_res_area)
            iou_txt.write("floor_areas: %.8f\n"%polygon_outline_area)

        return None
    
    


class Vector_output():
    def __init__(self, proj_name, roots, Shearwalls, Beams, bin_shearwall_img, archi_img, pixel2CAD_scale):
        self.raw_center_dir = os.path.join(roots[0], "raw_center_coords.txt")
        self.process_shearwall_img_dir = os.path.join(roots[3], proj_name+"_shearwall_bin_vect.png")
        self.output_shearwall_img_dir = os.path.join(roots[4], proj_name+"_archi_shearwall.png")
        self.output_beam_img_dir = os.path.join(roots[4], proj_name+"_archi_shearwall_beam.png")
        self.output_struct_img_dir = os.path.join(roots[4], proj_name+"_shearwall_beam.png")
        self.output_shearwall_txt_dir = os.path.join(roots[4], proj_name+"_shearwall_scaled.txt")
        self.output_beam_txt_dir = os.path.join(roots[4], proj_name+"_beam_scaled.txt")
        self.output_raw_shearwall_txt_dir = os.path.join(roots[4], proj_name+"_shearwall_raw.txt")
        self.output_raw_beam_txt_dir = os.path.join(roots[4], proj_name+"_beam_raw.txt")
        self.output_model_data_dir = os.path.join(roots[5], proj_name+"model_data.txt")
        self.bin_shearwall_img = bin_shearwall_img  # 只有剪力墙的二值图
        self.archi_img = archi_img  # 输入用的建筑图
        self.img_W = 512
        self.img_H = 1024
        self.canvas_struct = 255*np.ones((self.img_W, self.img_H, 3), np.uint8)  # 创建结构构件图空白画布
        # 剪力墙坐标数据
        self.Shearwall_xs, self.Shearwall_ys = Shearwalls[0], Shearwalls[1]
        # 梁坐标数据
        self.Beam_xs, self.Beam_ys = Beams[0], Beams[1]
        # 颜色BGR
        self.color_red = (0, 0, 255)  # 用于剪力墙
        self.color_yellow = (0, 255, 255)  # 用于梁
        # 像素图到真实坐标反向映射的scale
        self.pixel2CAD_scale = pixel2CAD_scale  # unit:pix/mm
        self.raw_center_coords = list(np.loadtxt(self.raw_center_dir))
    
    def img_shearwall_output(self):
        # 创建剪力墙二值图的画布
        img_W, img_H = self.bin_shearwall_img.shape[0], self.bin_shearwall_img.shape[1]
        # RGB_bin_img = np.zeros((img_W, img_H, 3), np.uint8)
        RGB_bin_img = 255*np.ones((img_W, img_H, 3), np.uint8)
        bin_shearwall_img_reverse = cv2.bitwise_not(src=self.bin_shearwall_img)
        RGB_bin_img[:, :, 0], RGB_bin_img[:, :, 1], RGB_bin_img[:, :, 2] = bin_shearwall_img_reverse, bin_shearwall_img_reverse, bin_shearwall_img_reverse  # 二值图转化为RGB
        
        # 逐构件输出
        for i, Shearwall_x in enumerate(self.Shearwall_xs):
            node1 = (Shearwall_x["x1"], Shearwall_x["y1"])
            node2 = (Shearwall_x["x2"], Shearwall_x["y2"])
            cv2.line(RGB_bin_img, node1, node2, self.color_red, 2)  # 在二值图上绘制
            cv2.line(self.archi_img, node1, node2, self.color_red, 6)  # 在输入的剪力墙图上绘制
            cv2.line(self.canvas_struct, node1, node2, self.color_red, 1)  # 在空白画布上绘制
            
        for i, Shearwall_y in enumerate(self.Shearwall_ys):
            node1 = (Shearwall_y["x1"], Shearwall_y["y1"])
            node2 = (Shearwall_y["x2"], Shearwall_y["y2"])
            cv2.line(RGB_bin_img, node1, node2, self.color_red, 2)  # 在二值图上绘制
            cv2.line(self.archi_img, node1, node2, self.color_red, 6)  # 在输入的剪力墙图上绘制
            cv2.line(self.canvas_struct, node1, node2, self.color_red, 1)  # 在空白画布上绘制
        
        # cv2.imshow('bin_vector', RGB_bin_img)
        # cv2.waitKey(0)
        cv2.imwrite(self.process_shearwall_img_dir, RGB_bin_img)
        cv2.imwrite(self.output_shearwall_img_dir, self.archi_img)
            
        return None
    
    def img_beam_output(self):
        # 逐构件输出
        for i, Beam_x in enumerate(self.Beam_xs):
            node1 = (Beam_x["x1"], Beam_x["y1"])
            node2 = (Beam_x["x2"], Beam_x["y2"])
            cv2.line(self.archi_img, node1, node2, self.color_yellow, 6)  # 在输入的剪力墙图上绘制
            cv2.line(self.canvas_struct, node1, node2, self.color_yellow, 1)  # 在空白画布上绘制
            
        for i, Beam_y in enumerate(self.Beam_ys):
            node1 = (Beam_y["x1"], Beam_y["y1"])
            node2 = (Beam_y["x2"], Beam_y["y2"])
            cv2.line(self.archi_img, node1, node2, self.color_yellow, 6)  # 在输入的剪力墙图上绘制
            cv2.line(self.canvas_struct, node1, node2, self.color_yellow, 1)  # 在空白画布上绘制

        cv2.imwrite(self.output_beam_img_dir, self.archi_img)
        cv2.imwrite(self.output_struct_img_dir, self.canvas_struct)
            
        return None
    
    def trans_coords(self, coords):
        trans_coords = []
        for coord in coords:
            node1_x = coord["x1"] - 0.5*self.img_H
            node1_y = coord["y1"] - 0.5*self.img_W
            node2_x = coord["x2"] - 0.5*self.img_H
            node2_y = coord["y2"] - 0.5*self.img_W
            trans_coords.append([node1_x, node1_y, node2_x, node2_y])
        
        return trans_coords
    
    def scale_coords(self, coords):
        scale_coords = []
        for coord in coords:
            node1_x = coord[0]*self.pixel2CAD_scale
            node1_y = coord[1]*self.pixel2CAD_scale
            node2_x = coord[2]*self.pixel2CAD_scale
            node2_y = coord[3]*self.pixel2CAD_scale
            scale_coords.append([node1_x, node1_y, node2_x, node2_y])
        
        return scale_coords
    
    def raw_vect_coords(self, coords):
        raw_center_x = self.raw_center_coords[0]
        raw_center_y = self.raw_center_coords[1]

        raw_vect_coords = []
        for coord in coords:
            node1_x = coord[0] + raw_center_x
            node1_y = coord[1] + raw_center_y
            node2_x = coord[2] + raw_center_x
            node2_y = coord[3] + raw_center_y
            raw_vect_coords.append([node1_x, node1_y, node2_x, node2_y])
        
        return raw_vect_coords
    
    def txt_output(self):
        # 移轴至中心
        self.trans_Shearwall_xs = self.trans_coords(self.Shearwall_xs)
        self.trans_Shearwall_ys = self.trans_coords(self.Shearwall_ys)
        self.trans_Beam_xs = self.trans_coords(self.Beam_xs)
        self.trans_Beam_ys = self.trans_coords(self.Beam_ys)
        
        # 缩放至正常水平
        self.scale_Shearwall_xs = self.scale_coords(self.trans_Shearwall_xs)
        self.scale_Shearwall_ys = self.scale_coords(self.trans_Shearwall_ys)
        self.scale_Beam_xs = self.scale_coords(self.trans_Beam_xs)
        self.scale_Beam_ys = self.scale_coords(self.trans_Beam_ys)
        
        scale_Shearwall_out = self.scale_Shearwall_xs + self.scale_Shearwall_ys
        scale_Beam_out = self.scale_Beam_xs + self.scale_Beam_ys
        
        # 移轴至原始CAD坐标处
        self.raw_Shearwall_xs = self.raw_vect_coords(self.scale_Shearwall_xs)
        self.raw_Shearwall_ys = self.raw_vect_coords(self.scale_Shearwall_ys)
        self.raw_Beam_xs = self.raw_vect_coords(self.scale_Beam_xs)
        self.raw_Beam_ys = self.raw_vect_coords(self.scale_Beam_ys)
        
        # 输出文本
        scale_Shearwall_out = self.scale_Shearwall_xs + self.scale_Shearwall_ys
        scale_Beam_out = self.scale_Beam_xs + self.scale_Beam_ys
        np.savetxt(self.output_shearwall_txt_dir, np.array(scale_Shearwall_out))  # 缩放至矢量尺寸的坐标
        np.savetxt(self.output_beam_txt_dir, np.array(scale_Beam_out))  # 缩放至矢量尺寸的坐标
        
        raw_Shearwall_out = self.raw_Shearwall_xs + self.raw_Shearwall_ys
        raw_Beam_out = self.raw_Beam_xs + self.raw_Beam_ys
        np.savetxt(self.output_raw_shearwall_txt_dir, np.array(raw_Shearwall_out))  # 缩放至矢量尺寸的坐标
        np.savetxt(self.output_raw_beam_txt_dir, np.array(raw_Beam_out))  # 缩放至矢量尺寸的坐标

        # 输出为建模数据
        output = open(self.output_model_data_dir, 'w')
        # output.write("剪力墙住宅建模数据，包括剪力墙坐标，连梁坐标和高度，框架梁坐标和高度，楼板坐标，单位均为：m\n\n")
        output.write("推荐使用中南建筑设计院 Swallow软件进行参数化建模\n")
        output.write("剪力墙住宅建模数据，包括剪力墙坐标，连梁坐标和高度，框架梁坐标和高度，楼板坐标，单位均为：m\n\n")
        output.write("Wall coordinates (point_i_x, point_i_y, point_j_x, point_j_y) num: %d \n"%len(scale_Shearwall_out))
        for wall in scale_Shearwall_out:
            output.write("%.3f, %.3f, %.3f, %.3f \n" % (wall[0] / 1000, wall[1] / 1000, wall[2] / 1000, wall[3] / 1000))

        frame_beams = []
        couple_beams = []
        for i, beam in enumerate(self.Beam_xs):
            if beam['type'] == 'frame':
                frame_beams.append(scale_Beam_out[i])
            elif beam['type'] == 'couple':
                couple_beams.append(scale_Beam_out[i])
            else:
                print("error!")
        for i, beam in enumerate(self.Beam_ys):
            if beam['type'] == 'frame':
                frame_beams.append(scale_Beam_out[len(self.Beam_xs) + i])
            elif beam['type'] == 'couple':
                couple_beams.append(scale_Beam_out[len(self.Beam_xs) + i])
            else:
                print("error!")
        output.write("\nCouplingBeam coordinates (point_i_x, point_i_y, point_j_x, point_j_y, height) num: %d \n"%len(couple_beams))
        for beam in couple_beams:
            length = ((beam[0] - beam[2]) ** 2 + (beam[1] - beam[3]) ** 2) ** 0.5 / 1000
            height = min(int((length * 0.4) / 0.1) * 0.1, 1.0)
            output.write("%.3f, %.3f, %.3f, %.3f, %.3f \n" % (beam[0] / 1000, beam[1] / 1000, beam[2] / 1000, beam[3] / 1000, height))
        output.write("\nFrameBeam coordinates (point_i_x, point_i_y, point_j_x, point_j_y, height) num: %d \n"%len(frame_beams))
        for beam in frame_beams:
            length = ((beam[0] - beam[2]) ** 2 + (beam[1] - beam[3]) ** 2) ** 0.5 / 1000
            height = min(max(int((length / 12) / 0.1) * 0.1, 0.4), 1.0)
            output.write("%.3f, %.3f, %.3f, %.3f, %.3f \n" % (beam[0] / 1000, beam[1] / 1000, beam[2] / 1000, beam[3] / 1000, height))
        output.close()
        
        return None
    

def main_pix_extract(proj_name, roots, pixel2CAD_scale):
    # 判断数据是否存在
    if not os.path.exists(roots[1]):
        print("Error: No structgan design %s" % proj_name)
    else:
        if not os.path.exists(roots[3]):
            os.makedirs(roots[3])  # 创建过程的图像文件夹
        if not os.path.exists(roots[4]):
            os.makedirs(roots[4])  # 创建结果的文件夹

        #### Step1 读取先验数据文件和StructGAN生成图像
        # 实例化
        files = Read_files(proj_name, roots)
        # 读取先验文件数据
        files.read_vectors()
        # 读取structGAN生成图像
        files.read_ganimg()
        
        #### Step2 提取structgan剪力墙并矢量化
        # 实例化
        gan_struct_extract = Structure_extract(proj_name, roots, files.archi_img, files.gan_shearwall_img,
                                              files.plot_Wall_coords, files.plot_Win_coords, files.plot_Gate_coords)
        # 剥离剪力墙像素
        gan_struct_extract.seperate_shearwall()
        # 提取剪力墙构件矢量
        gan_struct_extract.vect_structures()
        
        #### Step3 提取engineer剪力墙并矢量化
        # 实例化
        eng_struct_extract = Structure_extract(proj_name, roots, files.archi_img, files.engineer_shearwall_img,
                                              files.plot_Wall_coords, files.plot_Win_coords, files.plot_Gate_coords)
        # 剥离剪力墙像素
        eng_struct_extract.seperate_shearwall()
        # 提取剪力墙构件矢量
        eng_struct_extract.vect_structures()
        
        #### Step4 提取楼板轮廓
        # 实例化
        archi_extract = Archi_extract(proj_name, roots, files.archi_img, files.archi_dir)
        # 提取楼板轮廓
        archi_extract.vect_floor()
        
        #### Step5 对比二者一致性
        Eng_img = files.engineer_shearwall_img
        GAN_structs = gan_struct_extract.Shearwall_xs + gan_struct_extract.Shearwall_ys
        Eng_structs = eng_struct_extract.Shearwall_xs + eng_struct_extract.Shearwall_ys
        eval_byiou = Eval_byIoU(Eng_img,GAN_structs,Eng_structs,roots[4],proj_name)
        eval_byiou.iou_eval()
        
        #### Step6 计算cantilever面积
        eval_byfloor = Eval_byfloor(files.archi_img,pixel2CAD_scale,GAN_structs,
                                    archi_extract.outline_cnt_pix,roots[4],proj_name)
        eval_byfloor.floor_eval()
   
    return None


if __name__ == "__main__":
    proj_name = "layout"
    pixel2CAD_scale = 1 / 0.02  # 像素图到真实坐标反向映射的scale，unit:pix/mm
    conv_scale_data_root = ".\\2_conv_scale_data"  #os.path.join("2_conv_scale_data", proj_name)
    pixel_image_root = ".\\3_input_pixel_image" #os.path.join("3_input_pixel_image", proj_name)
    gan_image_root = ".\\4_structgan_gen_image" #os.path.join("4_structgan_gen_image", proj_name)
    process_image_root = ".\\5_process_image" #os.path.join("5_process_image", proj_name)
    eval_results_root = ".\\6_eval_results" #os.path.join("5_process_image", proj_name)

    main_pix_extract(proj_name, [conv_scale_data_root, pixel_image_root, gan_image_root, 
                                 process_image_root, eval_results_root], pixel2CAD_scale)