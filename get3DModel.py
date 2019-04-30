import cv2
import numpy as np
import random
import imutils
import imagePreProcessing
import getSubBunches
import getRadiusRangeManual
import myHoughCircle1
import myHoughCircle2
from myHoughCircle1 import myHoughCircle1
from myHoughCircle2 import myHoughCircle2
from imagePreProcessing import imagePreProcessing
from getSubBunches import getSubBunches
from getRadiusRangeManual import getRadiusRangeManual
color = 'p'
rgb = cv2.imread('6.jpg')
contours = imagePreProcessing(rgb,color)
subBunches = getSubBunches(rgb,contours)
def getSparseFactor(mask,bw,color):
    ret, th = cv2.threshold(bw, 0, 255, cv2.THRESH_OTSU)
    threshLevel = ret
    if color == 'p':
       bw_bunch_s = ~th
    elif color == 'g':
       bw_bunch_s = th
    else:
       print('wrong color!')
    sf = np.sum(bw_bunch_s)/np.sum(mask)
    return sf,bw_bunch_s
######################################
def get3DModel(subBunches,color):
    for i in range(1):
       if color == 'p':
          HSV = subBunches.rgb
          (h, s, v) = cv2.split(subBunches.rgb)
          bw = h
       elif color == 'g':
          lab = cv2.cvtColor(subBunches.rgb, cv2.COLOR_BGR2LAB)
          bw = lab[:, :, 2]
       else:
          print('wrong color!')
       sf,bw_bunch_s = getSparseFactor(subBunches.mask,bw,color)
       subBunches.sf = sf
       subBunches.bw_bunch_s = bw_bunch_s
##########################################
    random.seed()
    if subBunches.orientation > 90:
       mask = imutils.rotate_bound(subBunches.mask,-(180-subBunches.orientation))
       rgb = imutils.rotate_bound(subBunches.rgb,-(180-subBunches.orientation))
       dim = mask.shape
       data = np.empty([dim[0],dim[1],3])
       data[:,:,0] = mask
       data[:,:,1] = mask
       data[:,:,2] = mask
       mask1 = data.astype(np.uint8)
       rgb = rgb*mask1
       bw_bunch_s = imutils.rotate_bound(subBunches.bw_bunch_s,-(180-subBunches.orientation))
    else:
       mask = imutils.rotate(subBunches.mask,-(90-subBunches.orientation))
       rgb = imutils.rotate(subBunches.rgb,-(90-subBunches.orientation))
       dim = mask.shape
       data = np.empty([dim[0],dim[1],3])
       data[:,:,0] = mask
       data[:,:,1] = mask
       data[:,:,2] = mask
       mask1 = data.astype(np.uint8)
       rgb = rgb*mask1
       bw_bunch_s = imutils.rotate(subBunches.bw_bunch_s,-(90-subBunches.orientation))
    if color == 'p':
       hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
       (h, s, v) = cv2.split(hsv)
       channel = v
    elif color == 'g':
       (B,G,R) = cv2.split(rgb)
       channel = G
       lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
       bunch_b = lab[:, :, 2]
    else:
       print('wrong color!')
    mask = np.array(mask,np.uint8)
    contours,hierarch=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
          area = cv2.contourArea(contours[i])
          if area < 1000:
              cv2.drawContours(mask,[contours[i]],-1,0,-1)
    contours,hierarch=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    d = np.empty(len(contours))
    for i in range(len(contours)):
       d[i] = len(contours[i])
    s = np.shape(mask)
    max_d = max(d)
    h = np.argmax(d)
    boundmax = contours[h]
    Bw = np.zeros(s,dtype=np.uint8)
    cv2.drawContours(Bw,[contours[h]],-1,(255,255,255),1)
    rangeR = getRadiusRangeManual(rgb)
    sensitivity = 0.98
    edgeThreshold = 0.2*255
    centers_x,centers_y,radii = myHoughCircle1(Bw,mask,rangeR,sensitivity,edgeThreshold)
    if len(radii) == 0:
       existing_berries =[]
       newBerries_atEdge = []
       visibleBerries =[]
    else:
       Q1 = np.percentile(radii,25)
       Q3 = np.percentile(radii,75)
       Spread = 1.5*(Q3-Q1)
       MaxValue = Q3 + Spread
       MinValue = Q1 - Spread
       index = np.where((MinValue<radii)&(radii<MaxValue))
       centers_x = centers_x[index]
       centers_y = centers_y[index]
       radii = radii[index]
       newRangeR = range(int(min(radii)),int(max(radii)))
       len_rd = len(radii)
       a = centers_x
       b = centers_y
       c = np.zeros(len_rd,dtype=np.uint8)
       r = radii
       max_rd = max(radii)
       min_rd = min(radii)
       candidates = np.ones((len_rd,1),dtype=np.uint8)
       group = np.zeros((len_rd,1),dtype=np.uint8)
       tolerance = 9
       step_move = 0.5
       step_radii = 0.01
       groupNo = 1
       mark = False
       for i in range(len_rd):
           if group[i] == 0 and mark:
              groupNo = max(group) + 1
           elif group[i] > 0:
              groupNo = group[i]
           mark = False
           distance = ((a[i]-a)*(a[i]-a)+(b[i]-b)*(b[i]-b)+(r[i]-r)*(r[i]-r))**0.5
           index = np.where((distance > 0) & (distance < (r[i] + r - tolerance)))
           if np.sum(index)>0:
              currentGroups = group[index]
              index1 = np.where(currentGroups == 0)
              np.delete(currentGroups,index1)
              if np.sum(group[index])>0:
                 groupNo = min([groupNo, currentGroups.any()])
              group[i] = groupNo
              group[index] = groupNo
              for j in range(len(currentGroups)):
                  idx = group == currentGroups[j]
                  group[idx] = groupNo
              mark = True
       newBerries_atEdge = []
       group = np.ravel(group)
       for i in range(max(group)):
           #print(group)
           index = np.where(group == i)
           tmp_centers_x = centers_x[index]
           tmp_centers_y = centers_y[index]
           tmp_radii = radii[index]
           tmp_centers_x = np.sort(tmp_centers_x)
           tmp_centers_y = np.sort(tmp_centers_y) 
           tmp_radii = np.sort(tmp_radii) 
           if len(tmp_radii)%2 ==1:
               middle_berry_idx = int(len(tmp_radii)/2) 
               
               
           else:
               middle_berry_idx = len(tmp_radii)/2
           if len(tmp_radii)!=0:       
              middle_berry_idx=int(middle_berry_idx)
              tmp_centers_x = tmp_centers_x.reshape((-1,1)) 
              tmp_centers_y = tmp_centers_y.reshape((-1,1))  
              tmp_radii = tmp_radii.reshape((-1,1)) 
              group_berries = np.hstack((tmp_centers_x, tmp_centers_y, tmp_radii))
              candidates = np.zeros(len(tmp_radii),dtype=np.uint8)
           
              candidates[middle_berry_idx] = 1
   
           for j in range(len(tmp_radii)):
              if j != middle_berry_idx:
                 tmp_berries = [tmp_centers_x[candidates],tmp_centers_y[candidates],tmp_radii[candidates]]
                 while 1:
                     
                     distance = (((group_berries[j, 1] - tmp_berries[:,1])*(group_berries[j, 1] - tmp_berries[:,1])+(group_berries[j, 2]
                     - tmp_berries[:,2])*(group_berries[j, 2] - tmp_berries[:,2])
                     +(group_berries[j, 3] - tmp_berries[:,3])*(group_berries[j, 3] - tmp_berries[:,3]))**0.5)
                     index2 = (distance > 0)&(distance < (group_berries[j, 4] + tmp_berries[:, 4] - tolerance))
                     if np.sum(index2)==0:
                        candidates[j] = 1
           newBerries_atEdge = [[newBerries_atEdge],[tmp_centers_x,tmp_centers_y,tmp_radii]]
    sensitivity = 0.99 
    edgeThreshold = 0.1*255
    newRangeR = rangeR
    centers_x1 = centers_x.reshape((-1,1))
    centers_y1 = centers_y.reshape((-1,1))
    centers = np.hstack((centers_x1,centers_y1))
    cv2.imwrite('test5.jpg',mask)  
  
    Vcenters_x,Vcenters_y,Vradii = myHoughCircle2(channel,mask,newRangeR,sensitivity,edgeThreshold)
    if len(Vradii) == 0:
       visibleBerries = []
    else:
       Vcenters_x1 = Vcenters_x.reshape((-1,1))
       Vcenters_y1 = Vcenters_y.reshape((-1,1))
       Vradii1 = Vradii.reshape((-1,1))
       visibleBerries = np.empty((len(Vradii),5),dtype=np.uint8)
       visibleBerries = np.hstack((Vcenters_x1,Vcenters_y1,np.zeros((len(Vradii),1),dtype=np.uint8),Vradii1))
       candidates =  np.ones((len(Vradii),1),dtype=np.uint8)
       for i in range(len(Vradii)):
           distance = ((visibleBerries[i, 0] - centers[:,0])*(visibleBerries[i, 0] - centers[:,0])+(visibleBerries[i, 1] - centers[:,1])*(visibleBerries[i, 1] - centers[:,1]))**0.5
       #visibleBerries = [Vcenters_x,Vcenters_y,np.zeros((len(Vradii),1),dtype=np.uint8),Vradii]
       #print(distance)
           index = np.where((distance > 0)&(distance < visibleBerries[i, 3]))
           if np.sum(index)>0:
              candidates[i] = 0
       index = np.where(candidates==0)
       visibleBerries=np.delete(visibleBerries,index,0)


    contours,hierarch=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    parts = np.ones((mask.shape[0],1),dtype=np.uint8)
    bunch_ratio = 5/6
    x,y,w,h = cv2.boundingRect(contours[0])
    thres = w/h*h+y
    parts[1:int(thres)] = 0
    if len(Vradii)==0:
       existing_berries = newBerries_atEdge
    else:
       existing_berries = newBerries_atEdge
       print(existing_berries)
       for i in range(visibleBerries.shape[0]):
           center_x = visibleBerries[i, 0]
           center_y = visibleBerries[i, 1]
           idx = np.where(mask[int(center_y),:]==1)
           majorAxis = (max(idx[0]) - min(idx[0]) + 1)/2
           track_center = np.hstack((majorAxis+min(min(idx)), center_y, 0))
           if parts[int(center_y)] == 0:
              minorAxis = bunch_ratio*majorAxis
              visibleBerries[i, 2] = (abs((1-(center_x - track_center[0])**2/(majorAxis - visibleBerries[i,3])**2)*(minorAxis - visibleBerries[i,3])**2))**0.5+track_center[2]
           else:
              track_radius = majorAxis - visibleBerries[i,3]
              visibleBerries[i,3] = (abs(track_radius**2 - (center_x - track_center[0])**2))**0.5+track_center[2]
           while 1:
              distance = ((visibleBerries[i,0] - existing_berries[:,0])**2+(visibleBerries[i,1] - existing_berries[:,1])**2+(visibleBerries[i, 2] - existing_berries[:,2])**2)**0.5
              index = np.where((distance > 0)&(distance < (visibleBerries[i, 3] + existing_berries[:, 3] - tolerance)))
    #return existing_berries,newBerries_atEdge,visibleBerries
get3DModel(subBunches[0],color)
