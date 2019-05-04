from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import product, combinations
def plot3DBunchModel(existing_berries,c):
   colors1 = np.array([[25,0,51],[51,0,102],[102,0,204],[127,0,255], [178,102,255], [209,204,255]])/256
   cn = 256
   cm = colors1.shape[0]
   ct0 = np.linspace(0,1,cm)
#ct0 = ct0.reshape(-1,1)
   ct = np.linspace(0,1,cn)
#ct = ct.reshape(-1,1)
   cr = np.interp(ct,ct0,colors1[:,0])
   cr = cr.reshape(-1,1)
   cg = np.interp(ct,ct0,colors1[:,1])
   cg = cr.reshape(-1,1)
   cb = np.interp(ct,ct0,colors1[:,2])
   cb = cr.reshape(-1,1)
   u = np.linspace(0, 2 * np.pi, 100)
   v = np.linspace(0, np.pi, 100)
   if c == 'p':
      cap = np.hstack((cr,cg,cb))
   else:
      cap = np.hstack((cg,cb,cg))
   fig = plt.figure()
   ax = Axes3D(fig)
   ax = fig.add_subplot(111, projection='3d')
   for k in range(len(existing_berries)):
      x = existing_berries[k,3]*np.outer(np.cos(u), np.sin(v))+ existing_berries[k,0]
      y = existing_berries[k,3]*np.outer(np.sin(u), np.sin(v))+ existing_berries[k,1]
      z = existing_berries[k,3]*np.outer(np.ones(np.size(u)), np.cos(v))+ existing_berries[k,2]
      ax.plot_surface(x, y, z,  rstride=1, cstride=1, cmap=plt.get_cmap('coolwarm'))
   ax.view_init(elev=0,azim=0)
      #ax = fig.add_subplot(122, projection='3d')
      #ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
   plt.show()
