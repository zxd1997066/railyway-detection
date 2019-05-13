import os
from numpy import array
import numpy as np
import pylab as pl

# the path store the file
# 这里假设有一个文件叫做test.csv 存储的是我们要画的数据
oriPath = "test1.txt"


# 创建一个函数用来读取数据
def get_data(lines):

    sizeArry=[]

    for line in lines:
        line = line.replace("\n","")

        line = float(line)

        sizeArry.append(line)

    return array(sizeArry)
# 首先打开文件从文件中读取数据
f=open(oriPath)
Lenths = get_data(f.readlines())
def draw_hist(lenths):
    data = lenths 
    
    bins = np.linspace(min(data),2000,20)
    hist, bin_edges = np.histogram(data, 10)
    print(hist)
    print(bin_edges)
    print((bin_edges[1]+bin_edges[2])/2)
    print(bin_edges[len(bin_edges)-1])
    print(bin_edges[len(bin_edges)-1]-500)
    print(bin_edges[len(bin_edges)-1]-150)

    pl.hist(data,bins)

    pl.xlabel('distance')

    pl.ylabel('Number of occurences')

    pl.title('Frequency distribution of number of distances')

    pl.show()



draw_hist(Lenths)
