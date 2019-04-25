import numpy as np
import matplotlib.pyplot as plt
import cv2

opencv_image=cv2.imread("D:\pychar_projects\dnc-master\CVND_exercise\opencv3\CV_2\\02\\filter\opencv.png")
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(opencv_image,-1,kernel)
plt.subplot(121),plt.imshow(opencv_image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()