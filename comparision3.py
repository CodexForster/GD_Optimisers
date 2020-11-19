import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Results for shrinked dataset with 5 fold cross-validation on HerBo with mini-batch size of 128. 
# Training, testing: 2560 images, 640 images for 10 epochs:

# beta = 0.1
y11 = np.array([61.25, 62.97, 63.28, 62.97, 62.03, 61.09, 59.69, 58.75, 57.97, 57.19]) # Cross-validation fold 1
y12 = np.array([59.06, 60.47, 62.34, 61.41, 61.25, 61.09, 60.94, 60.0, 58.91, 58.59]) # Cross-validation fold 2
y13 = np.array([37.66, 34.84, 34.69, 33.59, 32.66, 32.03, 30.94, 30.47, 30.16, 29.84])
y14 = np.array([22.81, 19.53, 19.22, 19.69, 19.84, 20.47, 21.56, 22.5, 22.81, 23.28])
y15 = np.array([57.50, 62.03, 62.34, 61.88, 62.03, 61.41, 60.47, 60.47, 60.78, 60.0])
y1 = (y11 + y12 + y13 + y14 + y15)/5

# beta = 0.3
y31 = np.array([61.25, 62.97, 63.28, 62.97, 62.03, 61.09, 59.69, 58.75, 57.97, 57.19]) # Cross-validation fold 1
y32 = np.array([50.78, 55.94, 58.59, 59.22, 59.06, 59.06, 58.59, 58.43, 58.43, 57.81]) # Cross-validation fold 2
y33 = np.array([52.97, 58.13, 58.75, 58.75, 59.22, 58.75, 58.44, 58.28, 58.28, 57.97])
y34 = np.array([53.75, 55.47, 55.16, 54.38, 54.06, 53.28, 52.81, 52.19, 52.03, 51.41])
y35 = np.array([49.84, 52.19, 50.47, 49.22, 47.97, 46.72, 45.94, 44.53, 43.91, 44.06])
y3 = (y31 + y32 + y33 + y34 + y35)/5

# beta = 0.5
y51 = np.array([61.25, 62.97, 63.28, 62.97, 62.03, 61.09, 59.69, 58.75, 57.97, 57.19]) # Cross-validation fold 1
y52 = np.array([50.78, 55.94, 58.59, 59.22, 59.06, 59.06, 58.59, 58.44, 58.44, 57.81]) # Cross-validation fold 2
y53 = np.array([52.97, 58.13, 58.75, 58.75, 59.22, 58.75, 58.44, 58.28, 58.28, 57.97])
y54 = np.array([53.75, 55.47, 55.16, 54.38, 54.06, 53.28, 52.81, 52.19, 52.03, 51.41])
y55 = np.array([49.84, 52.19, 50.47, 49.22, 47.97, 46.72, 45.94, 44.53, 43.91, 44.06])
y5 = (y51 + y52 + y53 + y54 + y55)/5

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.xlabel('Number of epochs')
plt.ylabel('Cross-validated Testing accuracy (%)')
ax1.plot(x, y1, label='b = 0.1')
ax1.plot(x, y3, label='b = 0.3')
ax1.plot(x, y5, label='b = 0.5')
plt.legend(loc='best')
plt.title('Comparision of accuracy for different b in HerBo((Training size, testing size) = (2560, 640)), minibatch size = 128')
plt.show()