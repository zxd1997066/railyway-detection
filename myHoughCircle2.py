import cv2
import numpy as np

def myHoughCircle2(image,iclose,rangeR,sensitivity,edgeThreshold):
    try:
       circles1= cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,min(rangeR),param1=int(edgeThreshold),param2=18,minRadius=int(min(rangeR)),maxRadius=int(max(rangeR)))
       cx = np.empty(0)
       cy = np.empty(0)
       r = np.empty(0)
       img = cv2.imread('test5.jpg')
       for i in circles1[0,:,:]:
           cx = np.append(cx,i[0])
           cy = np.append(cy,i[1])
           r = np.append(r,i[2])
           #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
       #cv2.imwrite("test7.jpg", img)
       circle_number = len(r)
       centers_x = cx
       centers_y = cy
       radii = r
    except: 
       centers_x =[]
       centers_y =[]
       radii = []
       newRangeR =[]
    return centers_x,centers_y,radii
