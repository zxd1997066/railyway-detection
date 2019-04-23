import cv2
import numpy as np

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
def processingSubBunches(subBunches,color):
    for i in range(len(subBunches)):
       if color == 'p':
          HSV = subBunches[i].rgb
          (h, s, v) = cv2.split(subBunches[i].rgb)
          bw = h
       elif color == 'g':
          lab = cv2.cvtColor(subBunches[i].rgb, cv2.COLOR_BGR2LAB)
          bw = lab[:, :, 2]
       else:
          print('wrong color!')
       sf,bw_bunch_s = getSparseFactor(subBunches[i].mask,bw,color)
       subBunches[i].sf = sf
       subBunches[i].bw_bunch_s = bw_bunch_s
       existing_berries, newBerries_atEdge, visibleBerries = get3DModel(subBunches(i),color)
       subBunches[i].existing_berries = existing_berries
       subBunches[i].newBerries_atEdge = newBerries_atEdge
       subBunches[i].visibleBerries = visibleBerries
    return subBunches 
