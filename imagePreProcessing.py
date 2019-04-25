import cv2
import numpy as np

def imagePreProcessing(rgb,color):
    if color == 'p':
       hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
       (h, s, v) = cv2.split(hsv)
       bunch_s = v
       ret, th = cv2.threshold(bunch_s, 0, 255, cv2.THRESH_OTSU)
       threshLevel = ret
       bw = th
       se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6, 6))
       bw = ~cv2.dilate(bw, se)
       contours,hierarch=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       mask = np.zeros(bunch_s.shape) 
       for i in range(len(contours)):
          area = cv2.contourArea(contours[i])
          if area > bw.shape[0]*bw.shape[1]/6:
              cv2.drawContours(mask,[contours[i]],-1,(255,255,255),-1)
       for i in range(0,mask.shape[0]):
          for j in range(0,mask.shape[1]):
            if mask[i][j] == 255:
               mask[i][j] =1
       se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
       mask2 = cv2.dilate(mask,se)
       dim = mask2.shape
       data = np.empty([dim[0],dim[1],3])
       data[:,:,0] = mask2
       data[:,:,1] = mask2
       data[:,:,2] = mask2
       mask3 = data.astype(np.uint8)
       rgb = rgb*mask3
       (B,G,R) = cv2.split(rgb)
       bunch_r = R
       ret, th = cv2.threshold(bunch_r, 0, 255, cv2.THRESH_OTSU)#different
       bw = th

       contours,hierarch=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       for i in range(len(contours)):
          area = cv2.contourArea(contours[i])
          if area < 1000:
              cv2.drawContours(bw,[contours[i]],-1,0,-1)
       nonoise = bw
       bw1 = ~bw
       im_floodfill = bw.copy()
       h, w = bw1.shape[:2]
       mask = np.zeros((h+2, w+2), np.uint8)
       cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255),cv2.FLOODFILL_FIXED_RANGE)
       im_floodfill_inv = cv2.bitwise_not(im_floodfill)
       ifill = bw | im_floodfill_inv
       iopen = cv2.morphologyEx(ifill, cv2.MORPH_OPEN, se)
       contours,hierarch=cv2.findContours(iopen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       for i in range(len(contours)):
          area = cv2.contourArea(contours[i])
          if area < 2000:
              cv2.drawContours(iopen,[contours[i]],-1,0,-1)
       contours,hierarch=cv2.findContours(iopen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    elif color == 'g':
       lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
       bunch_b = lab[:, :, 2]
       ret, th = cv2.threshold(bunch_b, 0, 255, cv2.THRESH_OTSU)
       threshLevel = ret-0.1*255
       ret,thresh1 = cv2.threshold(bunch_b,threshLevel,255,cv2.THRESH_BINARY)
       bw = thresh1
       bunch_l = lab[:, :, 0]
       ret,thresh = cv2.threshold(bunch_l,0.2*255,255,cv2.THRESH_BINARY)
       bw = bw*thresh
       contours,hierarch=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       for i in range(len(contours)):
          area = cv2.contourArea(contours[i])
          if area < 1000:
              cv2.drawContours(bw,[contours[i]],-1,0,-1)
       nonoise = bw
       bw1 = ~bw
       im_floodfill = bw.copy()
       h, w = bw1.shape[:2]
       mask = np.zeros((h+2, w+2), np.uint8)
       cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255),cv2.FLOODFILL_FIXED_RANGE)
       im_floodfill_inv = cv2.bitwise_not(im_floodfill)
       ifill = bw | im_floodfill_inv
       se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
       iopen = cv2.morphologyEx(ifill, cv2.MORPH_CLOSE, se)
       contours,hierarch=cv2.findContours(iopen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       for i in range(len(contours)):
          area = cv2.contourArea(contours[i])
          if area < 2000:
              cv2.drawContours(iopen,[contours[i]],-1,0,-1)
       contours,hierarch=cv2.findContours(iopen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    else:
       print('wrong color!')
    return contours
