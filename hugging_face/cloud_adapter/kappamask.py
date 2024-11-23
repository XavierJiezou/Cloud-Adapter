# -*- coding: utf-8 -*-
# @Time    : 2024/8/7 下午3:51
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : kappamask.py.py
# @Software: PyCharm

import torch
from torch import nn as nn
from torch.nn import functional as F


class KappaMask(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.drop4 = nn.Dropout(0.5)

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.drop5 = nn.Dropout(0.5)

        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(1024, 512, 2),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(512, 256, 2),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(256, 128, 2),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.up9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(128, 64, 2),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Conv2d(2, num_classes, 1)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, 2, 2)

        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, 2, 2)

        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3, 2, 2)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = F.max_pool2d(drop4, 2, 2)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        merge6 = torch.cat((drop4, up6), dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = self.conv9(merge9)

        output = self.conv10(conv9)
        return output


if __name__ == '__main__':
    model = KappaMask(num_classes=2, in_channels=3)
    fake_data = torch.rand(2, 3, 256, 256)
    output = model(fake_data)
    print(output.shape)