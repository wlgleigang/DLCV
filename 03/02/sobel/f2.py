import cv2
import numpy as np
import matplotlib.pyplot as plt

gray=cv2.imread("D:\pychar_projects\dnc-master\CVND_exercise\opencv3\CV_2\\02\sobel\dave.png",0)
sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,5)
sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,5)
laplacian = cv2.Laplacian(gray, cv2.CV_64F, None, 5)
f,(ax1,ax2,ax3,ax4)=plt.subplots(1,4)
ax1.imshow(gray,'gray')
ax2.imshow(sobel_x,'gray')
ax3.imshow(sobel_y,'gray')
ax4.imshow(laplacian,"gray")

plt.show()