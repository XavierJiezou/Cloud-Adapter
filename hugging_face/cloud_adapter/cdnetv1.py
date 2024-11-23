# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 上午11:36
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : cdnetv1.py
# @Software: PyCharm

"""Cloud detection Network"""

"""Cloud detection Network"""

"""
This is the implementation of CDnetV1 without multi-scale inputs. This implementation uses ResNet by default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(_ASPP, self).__init__()
        out_channels = 512  # changed from 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        # self.project = nn.Sequential(
        # nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
        # norm_layer(out_channels),
        # nn.ReLU(True),
        # nn.Dropout(0.5))
        self.dropout2d = nn.Dropout2d(0.3)

    def forward(self, x):
        feat1 = self.dropout2d(self.b0(x))
        feat2 = self.dropout2d(self.b1(x))
        feat3 = self.dropout2d(self.b2(x))
        feat4 = self.dropout2d(self.b3(x))
        feat5 = self.dropout2d(self.b4(x))
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        # x = self.project(x)
        return x


class _FPM(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super(_FPM, self).__init__()
        self.aspp = _ASPP(in_channels, [6, 12, 18], norm_layer=norm_layer)
        # self.dropout2d = nn.Dropout2d(0.5)

    def forward(self, x):
        x = torch.cat((x, self.aspp(x)), dim=1)
        # x = self.dropout2d(x) # added
        return x


class BR(nn.Module):
    def __init__(self, num_classes, stride=1, downsample=None):
        super(BR, self).__init__()
        self.conv1 = conv3x3(num_classes, num_classes * 16, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_classes * 16, num_classes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out += residual

        return out


class CDnetV1(nn.Module):
    def __init__(self, in_channels=3,block=Bottleneck, layers=[3, 4, 6, 3], num_classes=21, aux=True):
        self.inplanes = 64
        self.aux = aux
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, affine = affine_par)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, affine=affine_par)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, affine=affine_par)

        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        self.res5_con1x1 = nn.Sequential(
            nn.Conv2d(1024 + 2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.fpm1 = _FPM(512, num_classes)
        self.fpm2 = _FPM(512, num_classes)
        self.fpm3 = _FPM(256, num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)

        self.predict1 = self._predict_layer(512 * 6, num_classes)
        self.predict2 = self._predict_layer(512 * 6, num_classes)
        self.predict3 = self._predict_layer(512 * 5 + 256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _predict_layer(self, in_channels, num_classes):
        return nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
                             nn.BatchNorm2d(256),
                             nn.ReLU(True),
                             nn.Dropout2d(0.1),
                             nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    # def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
    # return block(dilation_series,padding_series,num_classes)

    def base_forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        size_conv1 = x.size()[2:]
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        res2 = x
        x = self.layer2(x)
        res3 = x
        x = self.layer3(x)
        res4 = x
        x = self.layer4(x)
        x = self.res5_con1x1(torch.cat([x, res4], dim=1))

        return x, res3, res2, size_conv1

    def forward(self, x):
        size = x.size()[2:]
        score1, score2, score3, size_conv1 = self.base_forward(x)
        # outputs = list()
        score1 = self.fpm1(score1)
        score1 = self.predict1(score1)  # 1/8
        predict1 = score1
        score1 = self.br1(score1)

        score2 = self.fpm2(score2)
        score2 = self.predict2(score2)  # 1/8
        predict2 = score2

        # first fusion
        score2 = self.br2(score2) + score1
        score2 = self.br3(score2)

        score3 = self.fpm3(score3)
        score3 = self.predict3(score3)  # 1/4
        predict3 = score3
        score3 = self.br4(score3)

        # second fusion
        size_score3 = score3.size()[2:]
        score3 = score3 + F.interpolate(score2, size_score3, mode='bilinear', align_corners=True)
        score3 = self.br5(score3)

        # upsampling + BR
        score3 = F.interpolate(score3, size_conv1, mode='bilinear', align_corners=True)
        score3 = self.br6(score3)
        score3 = F.interpolate(score3, size, mode='bilinear', align_corners=True)
        score3 = self.br7(score3)

        # if self.aux:
        # auxout = self.dsn(mid)
        # auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
        # #outputs.append(auxout)
        return score3
        # return score3, predict1, predict2, predict3


if __name__ == '__main__':
    model = CDnetV1(num_classes=21)
    fake_image = torch.randn(2, 3, 224, 224)
    outputs = model(fake_image)
    for out in outputs:
        print(out.shape)
    # torch.Size([2, 21, 224, 224])
    # torch.Size([2, 21, 29, 29])
    # torch.Size([2, 21, 29, 29])
    # torch.Size([2, 21, 57, 57])