import cv2
import numpy as np

class Data:
    pass
def getSubBunches(rgb,contours):
    mask_len = len(contours)
    subBunch = [Data()]*mask_len
    for i in range(mask_len):
       subBunch[i].position = cv2.boundingRect(contours[i])
       subBunch[i].orientation = cv2.fitEllipse(contours[i])[2]
       (B,G,R) = cv2.split(rgb)
       subBunch[i].mask = np.zeros(B.shape)
       cv2.drawContours(subBunch[i].mask,[contours[i]],-1,(255,255,255),-1)
       subBunch[i].mask = subBunch[i].mask[int(subBunch[i].position[1]-1):int(subBunch[i].position[1]+subBunch[i].position[3]),int(subBunch[i].position[0]-1):int(subBunch[i].position[0]+subBunch[i].position[2])]
       subBunch[i].rgb = rgb[int(subBunch[i].position[1]-1):int(subBunch[i].position[1]+subBunch[i].position[3]),int(subBunch[i].position[0]-1):int(subBunch[i].position[0]+subBunch[i].position[2])]
    return subBunch


