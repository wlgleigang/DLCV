import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('D:/image/zhangmeng3.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])
print(hist,bins)
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
x = np.array([1, 2, 4, 7, 0])
y=np.diff(x)
z=np.diff(x, n=2)
print(x,y,z)
a = np.arange(5)
hist, bin_edges = np.histogram(a, density=True)
print(hist)
print(hist.sum())
print(np.sum(hist*np.diff(bin_edges)))

