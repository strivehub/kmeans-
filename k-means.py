import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_table("kmeans_data.txt",header=None,names=("x",'y'))
x = data['x']
y = data['y']

def centers(k):
    center = np.random.choice(np.arange(-5,5,0.1),size=[k,2])
    return center

def distence(data,center):
    dis = np.zeros(shape=[data.shape[0],center.shape[0]])
    for i in range(len(data)):
        for j in range(len(center)):
            dis[i][j] = np.sqrt(np.sum(np.square(data.iloc[i]-center[j])))
    return dis

def near_cen(dis):
    result = np.argmin(dis,axis=1)
    return result

def kmeans(data,k):
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