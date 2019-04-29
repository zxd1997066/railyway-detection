import cv2
import numpy as np
import random
import imutils
import imagePreProcessing
import getSubBunches
import getRadiusRangeManual
import myHoughCircle1
from myHoughCircle1 import myHoughCircle1
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
       print(subBunches.orientation)
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
       print(subBunches.orientation)
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
    cv2.imwrite('test5.jpg',mask)
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
           a[:] = a[:]-a[i]
           b[:] = b[:]-b[i]
           r[:] = r[:]-r[i]
           distance = (a*a+b*b+r*r)**0.5
           a = centers_x
           b = centers_y
           r = radii
           index = (distance > 0) & (distance < (r[i] + r - tolerance))
           index = index.astype(int)
           if np.sum(index)>0:
              currentGroups = group[index]
              index1 = np.where(currentGroups == 0)
              currentGroups[index1] = []
              if np.sum(group[index])>0:
                 groupNo = min([groupNo, currentGroups])
              group[i] = groupNo
              group[index] = groupNo
              for j in range(leng(currentGroups)):
                  idx = group == currentGroups[j]
                  group[idx] = groupNo
              mark = True
       newBerries_atEdge = []
       for i in range(max(group)[0]):
           index = np.where(group == i)
           tmp_centers_x = centers_x[index]
           tmp_centers_y = centers_y[index]
           tmp_radii = radii[index]
           tmp_centers_x = np.sort(tmp_centers_x)
           tmp_centers_y = np.sort(tmp_centers_y) 
           tmp_radii = np.sort(tmp_radii) 
           if len(tmp_radii)%2 ==1:
               middle_berry_idx = int(len(tmp_radii)/2)   
    #return existing_berries,newBerries_atEdge,visibleBerries
get3DModel(subBunches[0],color)
