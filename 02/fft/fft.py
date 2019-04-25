import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread("D:\pychar_projects\dnc-master\CVND_exercise\opencv3\CV_2\\02\\fft\opencv.png",0)
fft=np.fft.fft2(image)
fftshift=np.fft.fftshift(fft)
pin=20*np.log(np.abs(fftshift))
fft1=np.fft.ifftshift(fftshift)
image1=np.fft.ifft2(fft1)
plt.subplot(1,3,1),plt.imshow(image,'gray')
plt.subplot(1,3,2),plt.imshow(pin,'gray')
plt.subplot(1,3,3),plt.imshow(np.abs(image1),'gray')
plt.show()