[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# 脸部关键点检测

## 项目简介
在这个项目中，计算视觉和深度学习的技术结合在一起建立一个脸部关键点检测系统。脸部关键点包括眼睛，鼻子，嘴巴，脸轮廓。关键点检测可以应用在很多方面，比如脸部追踪，情绪检测等。
这个项目可以分为五个部分：
__Notebook 1__ : 加载和可视化数据，以及数据的处理

__Notebook 2__ : 定义卷积神经网络预测脸部关键点，以及相关训练

__Notebook 3__ : 使用Haar级联检测器检测人脸，以及使用训练好的网络做脸部关键点的预测

__Notebook 4__ : 脸部关键点应用1

__Notebook 5__ : 脸部关键点应用2

## 项目结构

### 搭建本地环境

1. 克隆文件(https://github.com/udacity/P1_Facial_Keypoints)

2. 建立一个Python 3.6的本地环境。

3. 安装pytorch。

4. 安装项目相关的包


### 数据

所有相关数据保存在data文件夹，文件夹有用于训练的图片和测试集的图片和关键点数据

### 未来改进
1.数据处理阶段的改进,增加旋转,图像金字塔等处理
2.尝试更多的卷积架构
3.
