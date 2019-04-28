import cv2
import numpy as np

def myHoughCircle1(image,iclose,rangeR,sensitivity,edgeThreshold):
  try:
    circles1= cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,int(sensitivity*min(rangeR)),param1=int(edgeThreshold),param2=7,minRadius=int(min(rangeR)),maxRadius=int(max(rangeR)))
    circles = np.single(np.around(circles1))
    cx = np.empty(0)
    cy = np.empty(0)
    r = np.empty(0)
    img = cv2.imread('test5.jpg')
    for i in circles1[0,:,:]:
        cx = np.append(cx,i[0])
        cy = np.append(cy,i[1])
        r = np.append(r,i[2])
    r_max = max(r)
    r_min = np.mean(r) + np.std(r, ddof = 1)*0.5
    newRangeR = range(int(r_min),int(r_max))
    centers_x = np.empty(0)
    centers_y = np.empty(0)
    radii = np.empty(0)
    for i in circles1[0,:,:]:
        if i[2] >= r_min and i[2] <= r_max:
           centers_x = np.append(centers_x,i[0])
           centers_y = np.append(centers_y,i[1])
           radii = np.append(radii,i[2])
    circle_number = len(radii)
    minDistance = int(r_min)
    circles2= cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,minDistance,param1=int(edgeThreshold),param2=7,minRadius=int(r_min),maxRadius=int(r_max))
    cx = np.empty(0)
    cy = np.empty(0)
    r = np.empty(0)
    for i in circles1[0,:,:]:
        cx = np.append(cx,i[0])
        cy = np.append(cy,i[1])
        r = np.append(r,i[2])
        #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    #cv2.imwrite("test7.jpg", img)
    centers_x = cx
    centers_y = cy
    radii = r
  except: 
    centers_x =[]
    centers_y =[]
    radii = []
    newRangeR =[]
  #print (centers_x,centers_y,radii)
  return centers_x,centers_y,radii

        


