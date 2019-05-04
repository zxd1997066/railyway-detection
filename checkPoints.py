import math
import numpy as np
def checkPoints(x,y,r,mask):
    r = r*0.8
    theta2 = [i for i in range(1,361,12)]
    xx = [(x+r*np.cos(theta2/180*np.pi)) for theta2 in theta2]
    yy = [(y+r*np.sin(theta2/180*np.pi)) for theta2 in theta2]
    xx = np.array(xx)
    xx = xx.reshape(-1,1)
    yy = np.array(yy)
    yy = yy.reshape(-1,1)
    ind1 = xx*yy > 0
    ind2 = xx<=mask.shape[0]
    ind3 = yy<= mask.shape[1]
    index4_1 = np.array_equal(ind1,ind2)
    index4_2 = np.array_equal(ind1,ind3)
    index4_3 = np.array_equal(ind2,ind3)
    vs = np.hstack((xx,yy))
    ind5 = 0
    index2 = False
    if index4_1 and index4_2 and index4_3:
      for xx,yy in vs:
         ind5 = ind5+mask[int(xx),int(yy)]
      if ind5==len(theta2):
         index2 = True
    return index2
