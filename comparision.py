import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Results for training, testing: 0:2560, 2560:5120 for 10 epochs:

#HerBo
y1 = np.array([45.66, 55.59, 58.32, 59.22, 59.73, 60.0, 60.35, 60.31, 60.23, 60.35])

#Plain SGD
y2 = np.array([70.59, 71.41, 71.52, 71.45, 71.48, 71.41, 71.48, 71.25])

#MBGD
y3 = np.array([46.25, 55.82, 58.87, 59.65, 60.39, 60.47, 60.78, 61.06, 61.21, 61.41])

#Adam mini-batch (both -ve updates)
#y4 = np.array([46.95, 52.31, 60.08, 65.20, 65.32]) old result for 5 epochs
y4 = np.array([46.52, 52.07, 60.55, 65.08, 65.31, 65.55, 65.82, 66.13, 66.48, 66.45])
#y4 = np.array([25.43, 22.42, 20.82, 20, 19.96])  #new dLdW update

#Adam SGD
y5 = np.array([67.77, 69.26, 70.08, 70.74, 71.09])

#Adam mini-batch (one +ve one -ve update)
y6 = np.array([26.52, 27.11, 29.53, 31.72, 34.65])
#y6 = np.array([62.42, 60.70, 55.66, 52.23, 0]) # 0 because it got interrupted. - new dLdW update

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Number of epochs')
plt.ylabel('Testing accuracy (%)')
ax1.plot(x, y1, label='HerBo with minibatch size 128')
ax1.plot(range(1,9), y2, label='Plain SGD')
ax1.plot(x, y3, label='MBGD')
ax1.plot(x, y4, label='Adam with minibatch size 128')
#ax1.plot(range(1,6), y5, label='Adam SGD')
#ax1.plot(x, y6, label='Adam with minibatch size 128')
plt.legend(loc='best')
plt.title('Comparision ((Training size, testing size) = (2560, 2560))')
plt.show()
