import cv2
import matplotlib.pyplot as plt
image_gray=cv2.imread("D:\pychar_projects\dnc-master\CVND_exercise\opencv3\CV_2\\02\clahe\\timg.jpg",0)
he=cv2.equalizeHist(image_gray)#自适应直方图制衡
clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))#限制对比度自适应直方图
cl1=clahe.apply(image_gray)
plt.subplot(131),plt.imshow(image_gray,'gray')
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(he,'gray')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(cl1,'gray')
plt.xticks([]),plt.yticks([])
plt.show()
