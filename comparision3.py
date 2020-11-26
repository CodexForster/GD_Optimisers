import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Results for shrinked dataset with 5 fold cross-validation on HerBo with mini-batch size of 128. 
# Training, testing: 2560 images, 640 images for 10 epochs:

# OLD beta = 0.1
y11 = np.array([61.25, 62.97, 63.28, 62.97, 62.03, 61.09, 59.69, 58.75, 57.97, 57.19]) # Cross-validation fold 1
y12 = np.array([59.06, 60.47, 62.34, 61.41, 61.25, 61.09, 60.94, 60.0, 58.91, 58.59]) # Cross-validation fold 2
y13 = np.array([37.66, 34.84, 34.69, 33.59, 32.66, 32.03, 30.94, 30.47, 30.16, 29.84]) # Cross-validation fold 3
y14 = np.array([22.81, 19.53, 19.22, 19.69, 19.84, 20.47, 21.56, 22.5, 22.81, 23.28]) # Cross-validation fold 4
y15 = np.array([57.50, 62.03, 62.34, 61.88, 62.03, 61.41, 60.47, 60.47, 60.78, 60.0]) # Cross-validation fold 5
y1 = (y11 + y12 + y13 + y14 + y15)/5

# beta = 0.001
y21 = np.array([61.25, 62.97, 63.28, 62.97, 62.03, 61.09, 59.69, 58.75, 57.97, 57.19]) # Cross-validation fold 1
y22 = np.array([50.78, 55.94, 58.59, 59.22, 59.06, 59.06, 58.59, 58.44, 58.44, 57.81]) # Cross-validation fold 2
y23 = np.array([52.97, 58.13, 58.75, 58.75, 59.22, 58.75, 58.44, 58.28, 58.28, 57.97]) # Cross-validation fold 3
y24 = np.array([53.75, 55.47, 55.16, 54.38, 54.06, 53.28, 52.81, 52.19, 52.03, 51.41]) # Cross-validation fold 4
y25 = np.array([49.84, 52.19, 50.47, 49.22, 47.97, 46.72, 45.94, 44.53, 43.91, 44.06]) # Cross-validation fold 5
y2 = (y21 + y22 + y23 + y24 + y25)/5

# beta = 0.01
y31 = np.array([61.25, 62.97, 63.28, 62.97, 62.03, 61.09, 59.69, 58.75, 57.97, 57.19]) # Cross-validation fold 1
y32 = np.array([50.78, 55.94, 58.59, 59.22, 59.06, 59.06, 58.59, 58.44, 58.44, 57.81]) # Cross-validation fold 2
y33 = np.array([52.97, 58.13, 58.75, 58.75, 59.22, 58.75, 58.44, 58.28, 58.28, 57.97]) # Cross-validation fold 3
y34 = np.array([53.75, 55.47, 55.16, 54.38, 54.06, 53.28, 52.81, 52.19, 52.03, 51.41]) # Cross-validation fold 4
y35 = np.array([49.84, 52.19, 50.47, 49.22, 47.97, 46.72, 45.94, 44.53, 43.91, 44.06]) # Cross-validation fold 5
y3 = (y31 + y32 + y33 + y34 + y35)/5

# beta = 5
y41 = np.array([61.41, 62.97, 63.44, 62.97, 62.19, 61.09, 59.8, 59.06, 57.97, 57.19]) # Cross-validation fold 1
y42 = np.array([50.79, 56.09, 58.75, 59.69, 59.06, 59.06, 58.75, 58.59, 58.44, 57.97]) # Cross-validation fold 2
y43 = np.array([52.97, 58.13, 59.22, 58.75, 59.22, 59.53, 59.06, 58.75, 58.28, 58.13]) # Cross-validation fold 3
y44 = np.array([53.75, 55.63, 55.0, 54.38, 54.22, 53.59, 52.97, 52.66, 52.19, 51.41]) # Cross-validation fold 4
y45 = np.array([50.31, 52.19, 50.16, 49.22, 47.97, 46.56, 45.94, 44.84, 44.06, 43.75 ]) # Cross-validation fold 5
y4 = (y41 + y42 + y43 + y44 + y45)/5

# beta = 7.5
y51 = np.array([61.41, 62.97, 63.44, 62.97, 62.34, 61.09, 60.16, 59.06, 58.13, 57.03]) # Cross-validation fold 1
y52 = np.array([50.79, 56.25, 58.75, 59.69, 59.06, 58.91, 58.44, 58.59, 58.44, 57.97]) # Cross-validation fold 2
y53 = np.array([52.97, 58.13, 59.38, 58.75, 59.34, 59.84, 59.06, 58.91, 58.28, 58.13]) # Cross-validation fold 3
y54 = np.array([53.91, 55.63, 55.0, 54.53, 54.38, 53.75, 53.13, 52.81, 52.50, 51.56]) # Cross-validation fold 4
y55 = np.array([50.31, 52.19, 50.16, 49.38, 47.97, 46.56, 45.78, 44.84, 44.22, 43.75]) # Cross-validation fold 5
y5 = (y51 + y52 + y53 + y54 + y55)/5

# beta = 10
y61 = np.array([61.41, 62.97, 63.28, 62.97, 62.5, 61.09, 60.16, 59.06, 58.28, 56.72]) # Cross-validation fold 1
y62 = np.array([50.79, 56.25, 58.75, 59.69, 59.38, 58.91, 58.44, 58.59, 58.44, 57.97]) # Cross-validation fold 2
y63 = np.array([52.97, 58.13, 59.34, 59.06, 59.69, 60.00, 59.22, 58.91, 58.59, 58.13]) # Cross-validation fold 3
y64 = np.array([54.06, 55.78, 55.0, 54.53, 54.69, 53.75, 53.43, 52.97, 52.50, 51.56]) # Cross-validation fold 4
y65 = np.array([50.31, 52.34, 50.16, 49.53, 47.97, 46.87, 45.47, 44.84, 44.37, 44.06]) # Cross-validation fold 5
y6 = (y61 + y62 + y63 + y64 + y65)/5

# beta = 12.5
y71 = np.array([61.41, 62.97, 63.28, 62.97, 62.5, 61.09, 60.47, 59.06, 58.28, 56.72]) # Cross-validation fold 1
y72 = np.array([50.79, 56.25, 58.75, 59.69, 59.38, 58.91, 58.44, 58.75, 58.28, 57.97]) # Cross-validation fold 2
y73 = np.array([53.13, 58.13, 59.69, 59.22, 60.00, 60.00, 59.38, 59.22, 58.75, 58.28]) # Cross-validation fold 3
y74 = np.array([54.06, 55.78, 55.16, 54.84, 54.69, 53.75, 53.75, 52.28, 52.81, 51.72]) # Cross-validation fold 4
y75 = np.array([50.63, 52.34, 50.16, 49.69, 48.13, 47.03, 45.47, 44.84, 44.53, 44.06]) # Cross-validation fold 5
y7 = (y71 + y72 + y73 + y74 + y75)/5

# beta = 15
y81 = np.array([61.41, 63.13, 63.75, 63.13, 62.5, 61.25, 60.47, 59.06, 58.28, 56.88]) # Cross-validation fold 1
y82 = np.array([50.79, 56.25, 58.75, 59.84, 59.53, 58.91, 58.44, 59.06, 58.28, 57.97]) # Cross-validation fold 2
y83 = np.array([53.13, 58.28, 59.84, 59.38, 60.16, 60.31, 59.69, 59.53, 58.91, 58.59]) # Cross-validation fold 3
y84 = np.array([54.06, 55.78, 55.00, 55.00, 55.00, 54.22, 54.06, 53.44, 52.97, 51.72]) # Cross-validation fold 4
y85 = np.array([50.78, 52.34, 50.31, 49.84, 48.28, 46.88, 45.31, 44.84, 44.69, 44.38]) # Cross-validation fold 5
y8 = (y81 + y82 + y83 + y84 + y85)/5

# beta = 20, crashes of 6 epochs
y91 = np.array([61.41, 63.13, 64.06, 63.13, 62.81, 61.41]) # Cross-validation fold 1
y92 = np.array([50.78, 56.41, 58.75, 59.69, 59.38, 59.38]) # Cross-validation fold 2
y93 = np.array([53.13, 58.59, 60.00, 60.47, 60.47, 60.47]) # Cross-validation fold 3
y94 = np.array([54.22, 55.63, 55.16, 55.47, 55.00, 54.69]) # Cross-validation fold 4
y95 = np.array([50.94, 52.34, 50.47, 49.69, 48.28, 47.03]) # Cross-validation fold 5
y9 = (y91 + y92 + y93 + y94 + y95)/5
y9_crash = [56.60, 56.60, 56.60, 56.60, 56.60]
x9_crash = [6, 7, 8, 9, 10]

# beta = 25, crashes after 2 epochs
y101 = np.array([61.56, 63.91]) # Cross-validation fold 1
y102 = np.array([51.09, 56.41]) # Cross-validation fold 2
y103 = np.array([53.13, 59.22]) # Cross-validation fold 3
y104 = np.array([54.53, 56.25]) # Cross-validation fold 4
y105 = np.array([50.78, 53.13]) # Cross-validation fold 5
y10 = (y101 + y102 + y103 + y104 + y105)/5
y10_crash = [57.78, 57.78, 57.78, 57.78, 57.78, 57.78, 57.78, 57.78, 57.78]
x10_crash = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Results for same train/test data 5-fold CV but using Vanilla MBGD
ym1 = np.array([61.72, 63.59, 66.09, 66.09, 65.94, 65.94, 66.25, 66.41, 66.25, 66.41]) # Cross-validation fold 1
ym2 = np.array([50.47, 56.56, 59.06, 60.16, 60.78, 60.94, 60.94, 64.47, 64.63, 60.94]) # Cross-validation fold 2
ym3 = np.array([52.97, 58.44, 59.69, 60.94, 61.25, 61.72, 61.56, 61.88, 62.19, 62.50]) # Cross-validation fold 3
ym4 = np.array([52.34, 55.00, 55.16, 55.16, 55.78, 55.63, 56.25, 56.41, 56.09, 55.94]) # Cross-validation fold 4
ym5 = np.array([50.31, 52.81, 52.03, 52.50, 51.88, 52.34, 52.19, 52.03, 52.03, 52.34]) # Cross-validation fold 5
ym = (ym1 + ym2 + ym3 + ym4 + ym5)/5

NUM_COLORS=12
cm = plt.get_cmap('gist_rainbow')
plt.rc('lines', linewidth=4)
plt.rc('axes', prop_cycle=(cycler(color=[cm(1.5*i/NUM_COLORS) for i in range(NUM_COLORS)])))
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Number of epochs')
plt.ylabel('Cross-validated Testing accuracy (%)')
ax1.set_prop_cycle(color=[cm(1.5*i/NUM_COLORS) for i in range(NUM_COLORS)], lw=1.7*np.ones(NUM_COLORS))
ax1.plot(x, ym, '#6D6D6D',label='Plain MBGD', linestyle='-.')
ax1.plot(range(1,3), y10, 'k', label='b = 25', linestyle='-.')
ax1.plot(range(1,7), y9, label='b = 20')
ax1.plot(x, y8, label='b = 15')
ax1.plot(x, y7, label='b = 12.5')
ax1.plot(x, y6, label='b = 10')
ax1.plot(x, y5, label='b = 7.5')
ax1.plot(x, y4, label='b = 5')
ax1.plot(x, y3, label='b = 0.01')
ax1.plot(x, y2, label='b = 0.001')
ax1.plot(x9_crash, y9_crash, 'r', linestyle='-')
ax1.plot(x10_crash, y10_crash, 'r', linestyle='-')
ax1.plot([2], [57.78], 'r', marker = 'x', label='Values shot up')
ax1.plot([6], [56.60], 'r', marker = 'x')
plt.legend(loc='best')
plt.title('Comparision Plot 3')
#plt.title('Comparision of accuracy for different b in HerBo, (Training size, testing size) = (2560, 640), minibatch size = 128')
plt.show()