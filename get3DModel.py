import cv2
import numpy as np
import random
import imutils
import imagePreProcessing
import getSubBunches
import getRadiusRangeManual
from imagePreProcessing import imagePreProcessing
from getSubBunches import getSubBunches
from getRadiusRangeManual import getRadiusRangeManual
color = 'g'
rgb = cv2.imread('4.jpg')
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
    if subBunches.orientation > 0:
       mask = imutils.rotate_bound(subBunches.mask,-(90-subBunches.orientation))
       rgb = imutils.rotate_bound(subBunches.rgb,-(90-subBunches.orientation))
       dim = mask.shape
       data = np.empty([dim[0],dim[1],3])
       data[:,:,0] = mask
       data[:,:,1] = mask
       data[:,:,2] = mask
       mask1 = data.astype(np.uint8)
       rgb = rgb*mask1
       bw_bunch_s = imutils.rotate_bound(subBunches.bw_bunch_s,-(90-subBunches.orientation))
    else:
       mask = imutils.rotate(subBunches.mask,-(90+subBunches.orientation))
       rgb = imutils.rotate(subBunches.rgb,-(90+subBunches.orientation))
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
    Bw = np.zeros(s)
    cv2.drawContours(Bw,[contours[h]],-1,(255,255,255),1)
    rangeR = getRadiusRangeManual(rgb)
    print(rangeR)
    cv2.imwrite('test3.jpg',Bw)
get3DModel(subBunches[0],color)
