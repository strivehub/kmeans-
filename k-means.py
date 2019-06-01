import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_table("kmeans_data.txt",header=None,names=("x",'y'))  #读取数据
x = data['x']
y = data['y']

def centers(k):  #随机产生中心的x,y坐标
    center = np.random.choice(np.arange(-5,5,0.1),size=[k,2])
    return center

def distence(data,center):  #计算每一个样本与K个中心坐标的欧氏距离，所以返回的是：样本数*坐标数维矩阵
    dis = np.zeros(shape=[data.shape[0],center.shape[0]])
    for i in range(len(data)):
        for j in range(len(center)):
            dis[i][j] = np.sqrt(np.sum(np.square(data.iloc[i]-center[j])))
    return dis

def near_cen(dis):   #返回每行与中心距离最近的索引
    result = np.argmin(dis,axis=1)
    return result

def kmeans(data,k):  #更新中心点坐标
    center = centers(k)
    for i in range(10):
        dis = distence(data, center)
        near_ = near_cen(dis)
        for j in range(k):
            center[j] = data[near_==j].mean()
    return center,near_

center,near_ = kmeans(data,4)

print(center)
plt.scatter(x,y,c=near_)
plt.scatter(center[:,0],center[:,1],c='red',s=500,marker="*")
plt.show()
