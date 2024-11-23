# -*- coding: utf-8 -*-
# @Time    : 2024/7/24 下午3:41
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : cdnetv2.py
# @Software: PyCharm

"""Cloud detection Network"""

"""
This is the implementation of CDnetV2 without multi-scale inputs. This implementation uses ResNet by default.
"""
# nn.GroupNorm

import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch import nn

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

        # self.layerx_1 = Bottleneck_nosample(64, 64, stride=1, dilation=1)
        # self.layerx_2 = Bottleneck(256, 64, stride=1, dilation=1, downsample=None)
        # self.layerx_3 = Bottleneck_downsample(256, 64, stride=2, dilation=1)


class Res_block_1(nn.Module):
    expansion = 4

    def __init__(self, inplanes=64, planes=64, stride=1, dilation=1):
        super(Res_block_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(8, planes),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                      padding=1, bias=False, dilation=1),
            nn.GroupNorm(8, planes),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.GroupNorm(8, planes * 4))

        self.relu = nn.ReLU(inplace=True)

        self.down_sample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4,
                      kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(8, planes * 4))

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        residual = self.down_sample(x)
        out += residual
        out = self.relu(out)

        return out


class Res_block_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes=256, planes=64, stride=1, dilation=1):
        super(Res_block_2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(8, planes),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                      padding=1, bias=False, dilation=1),
            nn.GroupNorm(8, planes),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.GroupNorm(8, planes * 4))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out


class Res_block_3(nn.Module):
    expansion = 4

    def __init__(self, inplanes=256, planes=64, stride=1, dilation=1):
        super(Res_block_3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(8, planes),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                      padding=1, bias=False, dilation=1),
            nn.GroupNorm(8, planes),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.GroupNorm(8, planes * 4))

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4,
                      kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(8, planes * 4))

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # residual = self.downsample(x)
        out += self.downsample(x)
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
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

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
        out_channels = 256
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

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class _DeepLabHead(nn.Module):
    def __init__(self, num_classes, c1_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)
        self.block = nn.Sequential(
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c1], dim=1))


class _CARM(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(_CARM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc1_2 = nn.Linear(in_planes // ratio, in_planes)

        self.fc2_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc2_2 = nn.Linear(in_planes // ratio, in_planes)
        self.relu = nn.ReLU(True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1_2(self.relu(self.fc1_1(avg_out)))

        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc2_2(self.relu(self.fc2_1(max_out)))

        max_out_size = max_out.size()[1]
        avg_out = torch.reshape(avg_out, (-1, max_out_size, 1, 1))
        max_out = torch.reshape(max_out, (-1, max_out_size, 1, 1))

        out = self.sigmoid(avg_out + max_out)

        x = out * x
        return x


class FSFB_CH(nn.Module):
    def __init__(self, in_planes, num, ratio=8):
        super(FSFB_CH, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc1_2 = nn.Linear(in_planes // ratio, num * in_planes)

        self.fc2_1 = nn.Linear(in_planes, in_planes // ratio)
        self.fc2_2 = nn.Linear(in_planes // ratio, num * in_planes)
        self.relu = nn.ReLU(True)

        self.fc3 = nn.Linear(num * in_planes, 2 * num * in_planes)
        self.fc4 = nn.Linear(2 * num * in_planes, 2 * num * in_planes)
        self.fc5 = nn.Linear(2 * num * in_planes, num * in_planes)

        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, num):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1_2(self.relu(self.fc1_1(avg_out)))

        max_out = self.max_pool(x)
        max_out = max_out.view(max_out.size(0), -1)
        max_out = self.fc2_2(self.relu(self.fc2_1(max_out)))

        out = avg_out + max_out
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.relu(self.fc5(out))  # (N, num*in_planes)

        out_size = out.size()[1]
        out = torch.reshape(out, (-1, out_size // num, 1, num))  # (N, in_planes, 1, num )
        out = self.softmax(out)

        channel_scale = torch.chunk(out, num, dim=3)  # (N, in_planes, 1, 1 )

        return channel_scale


class FSFB_SP(nn.Module):
    def __init__(self, num, norm_layer=nn.BatchNorm2d):
        super(FSFB_SP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 2 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(2 * num),
            nn.ReLU(True),
            nn.Conv2d(2 * num, 4 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(4 * num),
            nn.ReLU(True),
            nn.Conv2d(4 * num, 4 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(4 * num),
            nn.ReLU(True),
            nn.Conv2d(4 * num, 2 * num, kernel_size=3, padding=1, bias=False),
            norm_layer(2 * num),
            nn.ReLU(True),
            nn.Conv2d(2 * num, num, kernel_size=3, padding=1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, num):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.softmax(x)
        spatial_scale = torch.chunk(x, num, dim=1)
        return spatial_scale


##################################################################################################################


class _HFFM(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm2d):
        super(_HFFM, self).__init__()
        out_channels = 256
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
        self.carm = _CARM(in_channels)
        self.sa = FSFB_SP(4, norm_layer)
        self.ca = FSFB_CH(out_channels, 4, 8)

    def forward(self, x, num):
        x = self.carm(x)
        # feat1 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat = feat1 + feat2 + feat3 + feat4
        spatial_atten = self.sa(feat, num)
        channel_atten = self.ca(feat, num)

        feat_ca = channel_atten[0] * feat1 + channel_atten[1] * feat2 + channel_atten[2] * feat3 + channel_atten[
            3] * feat4
        feat_sa = spatial_atten[0] * feat1 + spatial_atten[1] * feat2 + spatial_atten[2] * feat3 + spatial_atten[
            3] * feat4
        feat_sa = feat_sa + feat_ca

        return feat_sa


class _AFFM(nn.Module):
    def __init__(self, in_channels=256, norm_layer=nn.BatchNorm2d):
        super(_AFFM, self).__init__()

        self.sa = FSFB_SP(2, norm_layer)
        self.ca = FSFB_CH(in_channels, 2, 8)
        self.carm = _CARM(in_channels)

    def forward(self, feat1, feat2, hffm, num):
        feat = feat1 + feat2
        spatial_atten = self.sa(feat, num)
        channel_atten = self.ca(feat, num)

        feat_ca = channel_atten[0] * feat1 + channel_atten[1] * feat2
        feat_sa = spatial_atten[0] * feat1 + spatial_atten[1] * feat2
        output = self.carm(feat_sa + feat_ca + hffm)
        # output = self.carm (feat_sa  + hffm)

        return output, channel_atten, spatial_atten


class block_Conv3x3(nn.Module):
    def __init__(self, in_channels):
        super(block_Conv3x3, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class CDnetV2(nn.Module):
    def __init__(self, in_channels=3,block=Bottleneck, layers=[3, 4, 6, 3], num_classes=21, aux=True):
        self.inplanes = 256  # change
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

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.3)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        # self.layer1 = self._make_layer(block, 64, layers[0])

        self.layerx_1 = Res_block_1(64, 64, stride=1, dilation=1)
        self.layerx_2 = Res_block_2(256, 64, stride=1, dilation=1)
        self.layerx_3 = Res_block_3(256, 64, stride=2, dilation=1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        self.hffm = _HFFM(2048, [6, 12, 18])
        self.affm_1 = _AFFM()
        self.affm_2 = _AFFM()
        self.affm_3 = _AFFM()
        self.affm_4 = _AFFM()
        self.carm = _CARM(256)

        self.con_layer1_1 = block_Conv3x3(256)
        self.con_res2 = block_Conv3x3(256)
        self.con_res3 = block_Conv3x3(512)
        self.con_res4 = block_Conv3x3(1024)
        self.con_res5 = block_Conv3x3(2048)

        self.dsn1 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.dsn2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

        # self.inplanes = 256 # change

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
        x = self.relu(self.bn1(self.conv1(x)))  # 1/2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)  # 1/4

        # x = self.layer1(x)	# 1/8

        # layer1
        x = self.layerx_1(x)  # 1/4
        layer1_0 = x

        x = self.layerx_2(x)  # 1/4
        layer1_0 = self.con_layer1_1(x + layer1_0)  # 256
        size_layer1_0 = layer1_0.size()[2:]

        x = self.layerx_3(x)  # 1/8
        res2 = self.con_res2(x)  # 256
        size_res2 = res2.size()[2:]

        # layer2-4
        x = self.layer2(x)  # 1/16
        res3 = self.con_res3(x)  # 256
        x = self.layer3(x)  # 1/16

        res4 = self.con_res4(x)  # 256
        x = self.layer4(x)  # 1/16
        res5 = self.con_res5(x)  # 256

        # x = self.res5_con1x1(torch.cat([x, res4], dim=1))
        return layer1_0, res2, res3, res4, res5, x, size_layer1_0, size_res2

        # return res2, res3, res4, res5, x, layer_1024,  size_res2

    def forward(self, x):
        # size = x.size()[2:]
        layer1_0, res2, res3, res4, res5, layer4, size_layer1_0, size_res2 = self.base_forward(x)

        hffm = self.hffm(layer4, 4)  # 256 HFFM
        res5 = res5 + hffm
        aux_feature = res5  # loss_aux
        # res5 = self.carm(res5)
        res5, _, _ = self.affm_1(res4, res5, hffm, 2)  # 1/16
        # aux_feature = res5
        res5, _, _ = self.affm_2(res3, res5, hffm, 2)  # 1/16

        res5 = F.interpolate(res5, size_res2, mode='bilinear', align_corners=True)
        res5, _, _ = self.affm_3(res2, res5, F.interpolate(hffm, size_res2, mode='bilinear', align_corners=True), 2)

        res5 = F.interpolate(res5, size_layer1_0, mode='bilinear', align_corners=True)
        res5, _, _ = self.affm_4(layer1_0, res5,
                                 F.interpolate(hffm, size_layer1_0, mode='bilinear', align_corners=True), 2)

        output = self.dsn1(res5)

        if self.aux:
            auxout = self.dsn2(aux_feature)
            # auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            # outputs.append(auxout)
        size = x.size()[2:]
        pred, pred_aux = output, auxout
        pred = F.interpolate(pred, size, mode='bilinear', align_corners=True)
        pred_aux = F.interpolate(pred_aux, size, mode='bilinear', align_corners=True)
        return pred
        return pred, pred_aux


if __name__ == '__main__':
    model = CDnetV2(num_classes=3)
    fake_image = torch.rand(2, 3, 256, 256)
    output = model(fake_image)
    for out in output:
        print(out.shape)
        # torch.Size([2, 3, 256, 256])
        # torch.Size([2, 3, 256, 256])