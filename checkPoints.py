import math
import numpy as np
def checkPoints(x,y,r,mask):
    r = r*0.8
    theta2 = [i for i in range(1,361,12)]
    xx = [(x+r*math.cos(theta2/180*math.pi)) for theta2 in theta2]
    yy = [(y+r*math.sin(theta2/180*math.pi)) for theta2 in theta2]
    xx = np.array(xx)
    xx = xx.reshape(-1,1)
    yy = np.array(yy)
    yy = yy.reshape(-1,1)
    vs = np.hstack((xx,yy))
    ind2 = 0
    for xx,yy in vs:
       if xx*yy > 0 and xx<=mask.shape[0] and yy<= mask.shape[1]:
          ind2 = ind2+mask[int(xx),int(yy)]
    if ind2==len(theta2):
          index2 = True
    else:
          index2 = False
    return index2
