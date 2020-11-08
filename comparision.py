import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])

#Plain SGD
y1 = np.array([70.59, 71.41, 71.52, 71.45, 71.48])

#MBGD
y2 = np.array([46.25, 55.82, 58.87, 59.65, 60.39])

#Adam SGD
y3 = np.array([67.77, 69.26, 70.08, 70.74, 71.09])

#Adam mini-batch
y4 = np.array([26.52, 27.11, 29.53, 31.72, 34.65])

#Adam mini-batch (both -ve updates)
y5 = np.array([46.95, 52.31, 60.08, 65.20, 65.32])


fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Number of epochs')
plt.ylabel('Testing accuracy (%)')
ax1.plot(x, y1, label='Plain SGD')
ax1.plot(x, y2, label='MBGD')
ax1.plot(x, y3, label='Adam SGD')
ax1.plot(x, y4, label='Adam with minibatch size 128')
ax1.plot(x, y5, label='Adam2 with minibatch size 128')
plt.legend(loc='best')
plt.title('Comparision ((Training size, testing size) = (2560, 2560))')
plt.show()
