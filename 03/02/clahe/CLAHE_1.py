import cv2
import matplotlib.pyplot as plt
import numpy as np

gray=cv2.imread("D:\pychar_projects\dnc-master\CVND_exercise\opencv3\CV_2\\02\clahe\\timg.jpg",0)
eh=cv2.equalizeHist(gray)
clane=cv2.createCLAHE(2,(10,10))
clane_gray=clane.apply(gray)
plt.subplot(1,3,1),plt.imshow(gray,'gray'),plt.xticks([]),plt.yticks([])
plt.subplot(1,3,2),plt.imshow(eh,'gray'),plt.xticks([]),plt.yticks([])
plt.subplot(1,3,3),plt.imshow(clane_gray,'gray'),plt.xticks([]),plt.yticks([])
plt.show()