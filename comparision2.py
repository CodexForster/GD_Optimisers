import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Results for shrinked dataset. Training, testing: 2560 images, 260 images for 10 epochs:

#HerBo
y1 = np.array([54.62, 56.92, 57.69, 57.69, 58.08, 58.85, 58.46, 57.69, 57.31, 55.77])

#Adam mini-batch
y2 = np.array([23.08, 28.08, 35.39, 38.08, 41.92, 46.15, 48.85, 50.0, 51.54, 52.70])

#Plain SGD
y3 = np.array([32.31, 36.54, 40.77, 42.69, 43.85, 45.77, 47.31, 47.31, 48.08])
y3_crash = np.array([48.08, 48.08])
x3_crash = np.array([9,10])

#MBGD
y4 = np.array([58.08, 60.39, 60.77, 60.39, 60.0, 60.39, 60.0, 60.0, 60.39, 60.39])

#Adam SGD
y5 = np.array([10.77, 7.31, 10.77, 10.77, 10.77, 10.77, 10.77, 10.77, 10.77, 10.77])

#Adam mini-batch (one +ve one -ve update)
y6 = np.array([26.52, 27.11, 29.53, 31.72, 34.65])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Number of epochs')
plt.ylabel('Testing accuracy (%)')
ax1.plot(x, y1, 'k', label='HerBo with minibatch size 128, b=0.1')
ax1.plot(range(1,10), y3, 'm', label='Plain SGD')
ax1.plot(x, y4, 'c', label='MBGD')
ax1.plot(x, y2, 'g', label='Adam with minibatch size 128')
ax1.plot(x, y5, 'y', label='Adam SGD')
ax1.plot([9],[48.08], 'r', marker='x', label='Values shot up')
ax1.plot(x3_crash, y3_crash, 'r')
plt.legend(loc='best')
plt.title('Comparision Plot 2')
#plt.title('Comparision ((Training size, testing size) = (2560, 260))')
plt.show()
