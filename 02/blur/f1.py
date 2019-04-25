import cv2
import matplotlib.pyplot as plt
import numpy as np
#读取灰度图
gray=cv2.imread("D:\pychar_projects\dnc-master\CVND_exercise\opencv3\CV_2\\02\\blur\opencv.png",0)
#在图片随机的位置点添加噪音
for i in range(2000):
    temp_x=np.random.randint(0,gray.shape[0])
    temp_y=np.random.randint(0,gray.shape[1])
    gray[temp_x,temp_y]=255
#比较过滤器去除噪点的效果
g_gray=cv2.GaussianBlur(gray,(5,5),0)
m_gray=cv2.medianBlur(gray,5)
plt.subplot(1,3,1),plt.imshow(gray,'gray')
plt.xticks([]),plt.yticks([])
plt.subplot(1,3,2),plt.imshow(g_gray,'gray')
plt.xticks([]),plt.yticks([])
plt.subplot(1,3,3),plt.imshow(m_gray,'gray')
plt.xticks([]),plt.yticks([])
plt.show()
