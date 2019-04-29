## 定义卷积模型架构
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torchvision import models
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #卷积层
        self.conv1=nn.Conv2d(1,32,5)
        self.conv2=nn.Conv2d(32,64,5)
        self.conv3=nn.Conv2d(64,128,5)
        self.conv4=nn.Conv2d(128,256,5)
        #批标准化层
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm1d(2048)
        self.bn6=nn.BatchNorm1d(1024)
        #池化层
        self.pool=nn.MaxPool2d(2,2)
        #dropout层
        self.drop1=nn.Dropout(p=0.1)
        self.drop2=nn.Dropout(p=0.2)
        self.drop3=nn.Dropout(p=0.3)
        self.drop4=nn.Dropout(p=0.4)
        self.drop5=nn.Dropout(p=0.5)
        self.drop6=nn.Dropout(p=0.6)
        #全连接层
        self.fc1=nn.Linear(256*10*10, 2048)
        self.fc2=nn.Linear(2048, 1024)
        self.fc3=nn.Linear(1024, 136)

    def forward(self, x):
        x=self.drop1(self.bn1(self.pool(F.relu(self.conv1(x)))))
        x=self.drop2(self.bn2(self.pool(F.relu(self.conv2(x)))))
        x=self.drop3(self.bn3(self.pool(F.relu(self.conv3(x)))))
        x=self.drop4(self.bn4(self.pool(F.relu(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x=self.drop5(self.bn5(self.fc1(x)))
        x=self.drop6(self.bn6(self.fc2(x)))
        x=self.fc3(x)
        return x


class vgg11_conv5_1(nn.Module):
    def __init__(self):
        super(vgg11_conv5_1, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)

        modules = list(vgg11.children())[:-3]
        modules[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.features = nn.Sequential(*modules)

        self.keypoints_estimator = nn.Sequential(
            nn.Linear(100352, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
