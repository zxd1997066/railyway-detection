import cv2
import imagePreProcessing
import getSubBunches
import processingSubBunches
import plot3DBunchModel
from plot3DBunchModel import plot3DBunchModel
from imagePreProcessing import imagePreProcessing
from getSubBunches import getSubBunches
from processingSubBunches import processingSubBunches
color = 'p'
rgb = cv2.imread('6.jpg')
contours = imagePreProcessing(rgb,color)
subBunches = getSubBunches(rgb, contours)
subBunches = processingSubBunches(subBunches,color)
subNum = len(subBunches)
berryNum = 0
berryNum_sf = 0

for k in range(subNum):
   berryNum = berryNum + subBunches[k].existing_berries.shape[0]
   berryNum_sf = berryNum_sf + subBunches[k].existing_berries.shape[0]*subBunches[k].sf
plot3DBunchModel(subBunches[0].existing_berries,color)
