# 论文地址：https://arxiv.org/abs/2407.07365
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
# BN_MOMENTUM = 0.01
relu_inplace = True
BN_MOMENTUM = 0.1
ALIGN_CORNERS = True

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


from yacs.config import CfgNode as CN
import math
from einops import rearrange

# configs for HRNet48
HRNET_48 = CN()
HRNET_48.FINAL_CONV_KERNEL = 1

HRNET_48.STAGE1 = CN()
HRNET_48.STAGE1.NUM_MODULES = 1
HRNET_48.STAGE1.NUM_BRANCHES = 1
HRNET_48.STAGE1.NUM_BLOCKS = [4]
HRNET_48.STAGE1.NUM_CHANNELS = [64]
HRNET_48.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_48.STAGE1.FUSE_METHOD = 'SUM'

HRNET_48.STAGE2 = CN()
HRNET_48.STAGE2.NUM_MODULES = 1
HRNET_48.STAGE2.NUM_BRANCHES = 2
HRNET_48.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_48.STAGE2.NUM_CHANNELS = [48, 96]
HRNET_48.STAGE2.BLOCK = 'BASIC'
HRNET_48.STAGE2.FUSE_METHOD = 'SUM'

HRNET_48.STAGE3 = CN()
HRNET_48.STAGE3.NUM_MODULES = 4
HRNET_48.STAGE3.NUM_BRANCHES = 3
HRNET_48.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_48.STAGE3.NUM_CHANNELS = [48, 96, 192]
HRNET_48.STAGE3.BLOCK = 'BASIC'
HRNET_48.STAGE3.FUSE_METHOD = 'SUM'

HRNET_48.STAGE4 = CN()
HRNET_48.STAGE4.NUM_MODULES = 3
HRNET_48.STAGE4.NUM_BRANCHES = 4
HRNET_48.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_48.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
HRNET_48.STAGE4.BLOCK = 'BASIC'
HRNET_48.STAGE4.FUSE_METHOD = 'SUM'

HRNET_32 = CN()
HRNET_32.FINAL_CONV_KERNEL = 1

HRNET_32.STAGE1 = CN()
HRNET_32.STAGE1.NUM_MODULES = 1
HRNET_32.STAGE1.NUM_BRANCHES = 1
HRNET_32.STAGE1.NUM_BLOCKS = [4]
HRNET_32.STAGE1.NUM_CHANNELS = [64]
HRNET_32.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_32.STAGE1.FUSE_METHOD = 'SUM'

HRNET_32.STAGE2 = CN()
HRNET_32.STAGE2.NUM_MODULES = 1
HRNET_32.STAGE2.NUM_BRANCHES = 2
HRNET_32.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_32.STAGE2.NUM_CHANNELS = [32, 64]
HRNET_32.STAGE2.BLOCK = 'BASIC'
HRNET_32.STAGE2.FUSE_METHOD = 'SUM'

HRNET_32.STAGE3 = CN()
HRNET_32.STAGE3.NUM_MODULES = 4
HRNET_32.STAGE3.NUM_BRANCHES = 3
HRNET_32.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_32.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRNET_32.STAGE3.BLOCK = 'BASIC'
HRNET_32.STAGE3.FUSE_METHOD = 'SUM'

HRNET_32.STAGE4 = CN()
HRNET_32.STAGE4.NUM_MODULES = 3
HRNET_32.STAGE4.NUM_BRANCHES = 4
HRNET_32.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_32.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRNET_32.STAGE4.BLOCK = 'BASIC'
HRNET_32.STAGE4.FUSE_METHOD = 'SUM'

HRNET_18 = CN()
HRNET_18.FINAL_CONV_KERNEL = 1

HRNET_18.STAGE1 = CN()
HRNET_18.STAGE1.NUM_MODULES = 1
HRNET_18.STAGE1.NUM_BRANCHES = 1
HRNET_18.STAGE1.NUM_BLOCKS = [4]
HRNET_18.STAGE1.NUM_CHANNELS = [64]
HRNET_18.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE1.FUSE_METHOD = 'SUM'

HRNET_18.STAGE2 = CN()
HRNET_18.STAGE2.NUM_MODULES = 1
HRNET_18.STAGE2.NUM_BRANCHES = 2
HRNET_18.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
HRNET_18.STAGE2.BLOCK = 'BASIC'
HRNET_18.STAGE2.FUSE_METHOD = 'SUM'

HRNET_18.STAGE3 = CN()
HRNET_18.STAGE3.NUM_MODULES = 4
HRNET_18.STAGE3.NUM_BRANCHES = 3
HRNET_18.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
HRNET_18.STAGE3.BLOCK = 'BASIC'
HRNET_18.STAGE3.FUSE_METHOD = 'SUM'

HRNET_18.STAGE4 = CN()
HRNET_18.STAGE4.NUM_MODULES = 3
HRNET_18.STAGE4.NUM_BRANCHES = 4
HRNET_18.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET_18.STAGE4.BLOCK = 'BASIC'
HRNET_18.STAGE4.FUSE_METHOD = 'SUM'


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
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
        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
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
            # att = self.downsample(att)
        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    # 创建平行层
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches  # 3
        num_inchannels = self.num_inchannels  # [48, 96, 192]
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRCloudNet(nn.Module):

    def __init__(self, in_channels=3,num_classes=2, base_c=48, **kwargs):
        global ALIGN_CORNERS
        extra = HRNET_48
        super(HRCloudNet, self).__init__()
        ALIGN_CORNERS = True
        # ALIGN_CORNERS = config.MODEL.ALIGN_CORNERS
        self.num_classes = num_classes
        # stem net
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)  # 只在pre[-1]与cur[-1]之间下采样？
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        self.out_conv = OutConv(base_c, num_classes)
        last_inp_channels = int(np.sum(pre_stage_channels))

        self.corr = Corr(nclass=2)
        self.proj = nn.Sequential(
            # 512 32
            nn.Conv2d(720, 48, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        # self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4, True)
        self.up3 = Up(base_c * 4, base_c * 2, True)
        self.up4 = Up(base_c * 2, base_c, True)
        fea_dim = 720
        bins = (1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    '''
    转换层的作用有两种情况：

    当前分支数小于之前分支数时，仅对前几个分支进行通道数调整。
    当前分支数大于之前分支数时，新建一些转换层，对多余的分支进行下采样，改变通道数以适应后续的连接。
    最终，这些转换层会被组合成一个 nn.ModuleList 对象，并在网络的构建过程中使用。
    这有助于确保每个分支的通道数在不同阶段之间能够正确匹配，以便进行特征的融合和连接
    '''

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        # 现在的分支数
        num_branches_cur = len(num_channels_cur_layer)  # 3
        # 处理前的分支数
        num_branches_pre = len(num_channels_pre_layer)  # 2

        transition_layers = []
        for i in range(num_branches_cur):
            # 如果当前分支数小于之前分支数，仅针对第一到第二阶段
            if i < num_branches_pre:
                # 如果对应层的通道数不一致，则进行转化（
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(

                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:  # 在新建层下采样改变通道数
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):  # 3
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    '''
    _make_layer 函数的主要作用是创建一个由多个相同类型的残差块（Residual Block）组成的层。
    '''

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    # 多尺度融合
    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):  # 重复4次
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, input, need_fp=True, use_corr=True):
        # from ipdb import set_trace
        # set_trace()
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        # x_176 = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):  # 2
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        # Y1
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        dict_return = {}
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)

        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        # x = self.stage3_(x)
        x[2] = self.up2(x[3], x[2])
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        # x = self.stage2_(x)
        x[1] = self.up3(x[2], x[1])
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x[0] = self.up4(x[1], x[0])
        xk = torch.cat([x[0], x1, x2, x3], 1)
        # PPM
        feat = self.ppm(xk)
        x = self.cls(feat)
        # fp分支
        if need_fp:
            logits = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
            # logits = self.out_conv(torch.cat((x, nn.Dropout2d(0.5)(x))))
            out = logits
            out_fp = logits
            if use_corr:
                proj_feats = self.proj(xk)
                corr_out = self.corr(proj_feats, out)
                corr_out = F.interpolate(corr_out, size=(352, 352), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp

            return dict_return['out']

        out = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if use_corr:  # True
            proj_feats = self.proj(xk)
            # 计算
            corr_out = self.corr(proj_feats, out)
            corr_out = F.interpolate(corr_out, size=(352, 352), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
        dict_return['out'] = out
        return dict_return['out']
        # return x

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(720, num_classes, kernel_size=1)
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels + out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Corr(nn.Module):
    def __init__(self, nclass=2):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(48, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(48, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):
        # in torch.Size([4, 32, 22, 22])
        # out = [4 2 352 352]
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        # out_temp 2 2 484
        # corr_map 4 484 484
        out = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)
        # out torch.Size([4, 2, 22, 22])
        return out


if __name__ == '__main__':
    input = torch.randn(4, 3, 352, 352)
    cloud = HRCloudNet(num_classes=2)
    output = cloud(input)
    print(output.shape)
    # torch.Size([4, 2, 352, 352]) torch.Size([4, 2, 352, 352]) torch.Size([4, 2, 352, 352])