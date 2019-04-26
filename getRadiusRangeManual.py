import cv2
import numpy as np

def getRadiusRangeManual(rgb):
    f= open("test1.txt","w+")
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
           f= open("test1.txt","a+")
           xy = "%d,%d" % (x, y)
           print(x,y,file=f)
           cv2.circle(rgb, (x, y), 1, (255, 0, 0), thickness = -1)
           cv2.putText(rgb, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0,0,0), thickness = 1)
           cv2.imshow("image", rgb)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    f.close
    data = np.loadtxt("test1.txt")
    data1=data.reshape(1,1,4)
    for i in range(0,data1.shape[0]):
      for x1,y1,x2,y2 in data1[i]:
        R = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))**0.5
    rangeR = range(int(R),int(R+10))
    return rangeR

