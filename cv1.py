import os
import cv2
import numpy as np
from numpy import array
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from pandas import Series
 
videos_src_path = "/home/eastward/ug thesis/opencv project/"
video_formats = [".MP4", ".MOV"]
frames_save_path = "/home/eastward/ug thesis/opencv project/cv5/"
width = 1920
height = 1080
time_interval = 1
 
def threshold_cluster(Data_set,threshold):
    stand_array=np.asarray(Data_set).ravel('C')
    stand_Data=Series(stand_array)
    index_list,class_k=[],[]
    while stand_Data.any():
        if len(stand_Data)==1:
            index_list.append(list(stand_Data.index))
            class_k.append(list(stand_Data))
            stand_Data=stand_Data.drop(stand_Data.index)
        else:
            class_data_index=stand_Data.index[0]
            class_data=stand_Data[class_data_index]
            stand_Data=stand_Data.drop(class_data_index)
            if (abs(stand_Data-class_data)<=threshold).any():
                args_data=stand_Data[abs(stand_Data-class_data)<=threshold]
                stand_Data=stand_Data.drop(args_data.index)
                index_list.append([class_data_index]+list(args_data.index))
                class_k.append([class_data]+list(args_data))
            else:
                index_list.append([class_data_index])
                class_k.append([class_data])
    return index_list,class_k 
def video2frame(video_src_path, formats, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param formats:　包含的所有视频格式
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    videos = os.listdir(video_src_path)
 
    def filter_format(x, all_formats):
        if x[-4:] in all_formats:
            return True
        else:
            return False
 
    videos = filter(lambda x: filter_format(x, formats), videos)
 
    for each_video in videos:
        print ("正在读取视频："), each_video
 
        each_video_name = each_video[:-4]
        os.mkdir(frame_save_path + each_video_name)
        each_video_save_full_path = os.path.join(frame_save_path, each_video_name) + "/"
        os.mkdir(each_video_save_full_path + each_video_name)
        each_video_save_full = os.path.join(frame_save_path) + "/"
        each_video_full_path = os.path.join(video_src_path, each_video)
        each_video_save_final_path = os.path.join(each_video_save_full_path, each_video_name) + "/"

        cap = cv2.VideoCapture(each_video_full_path)
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            print("读取失败!")
 
        while(success):
            success, frame = cap.read()
            
 
            if frame_index % interval == 0:
                resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(resize_frame,cv2.COLOR_BGR2GRAY)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
                cv2.imwrite(each_video_save_full + "%d.pgm" % frame_count, gray)
                cv2.imwrite(each_video_save_full + "%d.jpg" % frame_count, resize_frame)
                prossed1_frame = os.system("./elsd" + " " + "%d.pgm" % frame_count)
                
                path = each_video_save_full_path+"%d.pgm" % frame_count
                path1 = each_video_save_full + "%d.jpg" % frame_count
                img = cv2.imread(path)
                imag = cv2.imread(path1)
                data = np.loadtxt("ellipses.txt")
                f= open("test1.txt","w+")
                f1= open("test2.txt","w+")
                a = data.shape[0]
                data1=data.reshape(a,1,4)
                data2=data1.astype(np.int)
                data3=np.empty([0,1,4])
                for i in range(0,data1.shape[0]):
                  for x1,y1,x2,y2 in data1[i]:
                    d = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))**0.5
                    print(d,file=f)
                f.close
                oriPath = "test1.txt"
                oriPath1 = "test2.txt"
                def get_data(lines):
                   sizeArry=[]
                   for line in lines:
                      line = line.replace("\n","")
                      line = float(line)
                      sizeArry.append(line)
                   return array(sizeArry)
                f=open(oriPath)
                Lenths = get_data(f.readlines())
                hist, bin_edges = np.histogram(Lenths,10)
                for i in range(0,data2.shape[0]):
                  for x1,y1,x2,y2 in data2[i]:
                     d = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))**0.5
                     if d > (bin_edges[1]+bin_edges[2])/2 and (y2-y1)/(x2-x1)>0:
                       print(x1,file=f1)
                       data3 = np.append(data3,[[[x1,y1,x2,y2]]], axis = 0)
                f1.close
                f1=open(oriPath1)
                Lenths1 = get_data(f1.readlines())
                Lenths1=sorted(Lenths1)
                index_list,class_k=threshold_cluster(Lenths1,450)
                a=len(class_k)
                td1=class_k[int(a)-1]
                td2=class_k[int(a)-2]
                if np.var(td1)<np.var(td2):
                   Lenths1 = td2
                else:
                   Lenths1 = td1
     
                hist, bin_edges = np.histogram(Lenths1,bins=2)
                
                break_flag=False
                for i in range(0,data3.shape[0]):
                   for x1,y1,x2,y2 in data3[i]:
                      if x1>=bin_edges[len(bin_edges)-1] and x1 in Lenths1:
                         break_flag=True
                         break
                   if break_flag==True:
                       break
                k = (y2-y1)/(x2-x1)
                b = y1 - k*x1
                cv2.line(imag,(int((0-b)/k),0),(int((1080-b)/k),1080),(0,0,255),3)
                break_flag=False
                for i in range(0,data3.shape[0]):
                   for x1,y1,x2,y2 in data3[i]:
                      if x1>bin_edges[len(bin_edges)-1]-500 and x1<bin_edges[len(bin_edges)-1]-150 and x1 in Lenths1:
                         break_flag=True
                         break
                   if break_flag==True:
                       break
                k = (y2-y1)/(x2-x1)
                b = y1 - k*x1
                cv2.line(imag,(int((0-b)/k),0),(int((1080-b)/k),1080),(0,0,255),3)
                cv2.imwrite(each_video_save_final_path+"%d.jpg" % frame_count,imag)
                frame_count += 1
 
            frame_index += 1
 
    cap.release()
 
 
if __name__ == '__main__':
    video2frame(videos_src_path, video_formats, frames_save_path, width, height, time_interval)
