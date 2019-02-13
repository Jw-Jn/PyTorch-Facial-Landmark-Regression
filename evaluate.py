import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pandas as pd

img_size = 225
max_epochs = 80
learning_rate = 0.0001
pretrained = True
str_pre = 'pre'
file_name = 'lfw_alexnet_'+str(learning_rate)+'_'+str(max_epochs)+'_'+str_pre

df = pd.read_csv(file_name+'.csv')
data = df.values

index = np.arange(0, 0.4, 0.01)
distance = []
for i in range(len(data)):
    if (i+1)%2 == 0: # even
        continue
    else:
        for j in range(7):
            dis = np.sqrt(pow((data[i][j*2]-data[i+1][j*2]), 2) 
            + pow((data[i][j*2+1]-data[i+1][j*2+1]), 2))
            distance.append(dis)
distance = np.asarray(distance)

count = []
for i in index:
    count.append(sum(distance<i) / len(distance) * 1.0)


plt.plot(index, count)
plt.xlabel('radius')
plt.ylabel('accuracy')
plt.savefig(file_name+'_radius.jpg')
# plt.show()