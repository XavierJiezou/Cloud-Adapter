# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 下午3:51
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : mcdnet.py
# @Software: PyCharm
import image_dehazer
import numpy as np
# 论文地址：https://www.sciencedirect.com/science/article/pii/S1569843224001742?via%3Dihub
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DPFF(nn.Module):
    def __init__(self, in_channels) -> None:
        super(_DPFF, self).__init__()
        self.cbr1 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, bias=False)
        self.cbr2 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.cbr3 = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.cbr4 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, bias=False)

    def forward(self, feature1, feature2):
        d1 = torch.abs(feature1 - feature2)
        d2 = self.cbr1(torch.cat([feature1, feature2], dim=1))
        d = torch.cat([d1, d2], dim=1)
        d = self.cbr2(d)
        # d = self.sigmoid(d)

        v1, v2 = self.cbr3(feature1), self.cbr3(feature2)
        v1, v2 = v1 * d, v2 * d
        features = torch.cat([v1, v2], dim=1)
        features = self.cbr4(features)

        return features


class DPFF(nn.Module):
    def __init__(self, layer_channels) -> None:
        super(DPFF, self).__init__()
        self.cfes = nn.ModuleList()
        for layer_channel in layer_channels:
            self.cfes.append(_DPFF(layer_channel))

    def forward(self, features1, features2):
        outputs = []
        for feature1, feature2, cfe in zip(features1, features2, self.cfes):
            outputs.append(cfe(feature1, feature2))
        return outputs


class DirectDPFF(nn.Module):
    def __init__(self, layer_channels) -> None:
        super(DirectDPFF, self).__init__()
        self.fusions = nn.ModuleList(
            [nn.Conv2d(layer_channel * 2, layer_channel, 1, 1) for layer_channel in layer_channels]
        )

    def forward(self, features1, features2):
        outputs = []
        for feature1, feature2, fusion in zip(features1, features2, self.fusions):
            feature = torch.cat([feature1, feature2], dim=1)
            outputs.append(fusion(feature))
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True,
                 bn=False, activation=True, maxpool=True):
        super(ConvBlock, self).__init__()
        self.module = []
        if maxpool:
            down = nn.Sequential(
                *[
                    nn.MaxPool2d(2),
                    nn.Conv2d(input_size, output_size, 1, 1, 0, bias=bias)
                ]
            )
        else:
            down = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.module.append(down)
        if bn:
            self.module.append(nn.BatchNorm2d(output_size))
        if activation:
            self.module.append(nn.PReLU())
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        out = self.module(x)

        return out


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True,
                 bn=False, activation=True, bilinear=True):
        super(DeconvBlock, self).__init__()
        self.module = []
        if bilinear:
            deconv = nn.Sequential(
                *[
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(input_size, output_size, 1, 1, 0, bias=bias)
                ]
            )
        else:
            deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.module.append(deconv)
        if bn:
            self.module.append(nn.BatchNorm2d(output_size))
        if activation:
            self.module.append(nn.PReLU())
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        out = self.module(x)

        return out


class FusionBlock(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, maxpool=False,
                 bilinear=False):
        super(FusionBlock, self).__init__()
        self.num_ft = num_ft
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), kernel_size, stride, padding,
                            bias=bias, bilinear=bilinear)
            )
            self.down_convs.append(
                ConvBlock(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size, stride, padding, bias=bias,
                          maxpool=maxpool)
            )

    def forward(self, ft_l, ft_h_list):
        ft_fusion = ft_l
        for i in range(len(ft_h_list)):
            ft = ft_fusion
            for j in range(self.num_ft - i):
                ft = self.up_convs[j](ft)
            ft = ft - ft_h_list[i]
            for j in range(self.num_ft - i):
                ft = self.down_convs[self.num_ft - i - j - 1](ft)
            ft_fusion = ft_fusion + ft

        return ft_fusion


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class AddRelu(nn.Module):
    """It is for adding two feed forwards to the output of the two following conv layers in expanding path
    """

    def __init__(self) -> None:
        super(AddRelu, self).__init__()
        self.relu = nn.PReLU()

    def forward(self, input_tensor1, input_tensor2, input_tensor3):
        x = input_tensor1 + input_tensor2 + input_tensor3
        return self.relu(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(BasicBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = ConvLayer(in_channels, mid_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_channels, momentum=0.1)
        self.relu = nn.PReLU()

        self.conv2 = ConvLayer(mid_channels, out_channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)

        self.conv3 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.conv3(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)

        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)

        self.conv3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=0.1)

        self.conv4 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)

        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.conv4(x)

        out = out + residual
        out = self.relu(out)

        return out


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PPM, self).__init__()

        self.pool_sizes = [1, 2, 3, 6]  # subregion size in each level
        self.num_levels = len(self.pool_sizes)  # number of pyramid levels

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_levels):
            self.conv_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=self.pool_sizes[i]),
                nn.Conv2d(in_channels, in_channels // self.num_levels, kernel_size=1),
                nn.BatchNorm2d(in_channels // self.num_levels),
                nn.ReLU(inplace=True)
            ))
        self.out_conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        input_size = x.size()[2:]  # get input size
        output = [x]

        # pyramid pooling
        for i in range(self.num_levels):
            out = self.conv_layers[i](x)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            output.append(out)

        # concatenate features from different levels
        output = torch.cat(output, dim=1)
        output = self.out_conv(output)

        return output


class MCDNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, maxpool=False, bilinear=False) -> None:
        super().__init__()
        level = 1
        # encoder
        self.conv_input = ConvLayer(in_channels, 32 * level, kernel_size=3, stride=2)

        self.dense0 = BasicBlock(32 * level, 32 * level)
        self.conv2x = ConvLayer(32 * level, 64 * level, kernel_size=3, stride=2)

        self.dense1 = BasicBlock(64 * level, 64 * level)
        self.conv4x = ConvLayer(64 * level, 128 * level, kernel_size=3, stride=2)

        self.dense2 = BasicBlock(128 * level, 128 * level)
        self.conv8x = ConvLayer(128 * level, 256 * level, kernel_size=3, stride=2)

        self.dense3 = BasicBlock(256 * level, 256 * level)
        self.conv16x = ConvLayer(256 * level, 512 * level, kernel_size=3, stride=2)

        self.dense4 = PPM(512 * level, 512 * level)

        # dpff
        self.dpffm = DPFF([32, 64, 128, 256, 512])

        # decoder
        self.convd16x = UpsampleConvLayer(512 * level, 256 * level, kernel_size=3, stride=2)
        self.fusion4 = FusionBlock(256 * level, 3, maxpool=maxpool, bilinear=bilinear)
        self.dense_4 = Bottleneck(512 * level, 256 * level)
        self.add_block4 = AddRelu()

        self.convd8x = UpsampleConvLayer(256 * level, 128 * level, kernel_size=3, stride=2)
        self.fusion3 = FusionBlock(128 * level, 2, maxpool=maxpool, bilinear=bilinear)
        self.dense_3 = Bottleneck(256 * level, 128 * level)
        self.add_block3 = AddRelu()

        self.convd4x = UpsampleConvLayer(128 * level, 64 * level, kernel_size=3, stride=2)
        self.fusion2 = FusionBlock(64 * level, 1, maxpool=maxpool, bilinear=bilinear)
        self.dense_2 = Bottleneck(128 * level, 64 * level)
        self.add_block2 = AddRelu()

        self.convd2x = UpsampleConvLayer(64 * level, 32 * level, kernel_size=3, stride=2)
        self.dense_1 = Bottleneck(64 * level, 32 * level)
        self.add_block1 = AddRelu()

        self.head = UpsampleConvLayer(32 * level, num_classes, kernel_size=3, stride=2)
        self.apply(self._weights_init)
    
    @torch.no_grad()
    def get_lr_data(self, x: torch.Tensor) -> torch.Tensor:
        images = x.cpu().permute(0, 2, 3, 1).numpy() # b, h, w, c
        batch_size = images.shape[0]
        lr = []
        for i in range(batch_size):
            lr_image = image_dehazer.remove_haze((images[i]*255).astype(np.uint8), showHazeTransmissionMap=False)[0] # h, w, c, numpy.array
            lr_tensor = torch.from_numpy(lr_image).permute(2, 0, 1)/255. # c, h, w
            lr.append(lr_tensor)
        return torch.stack(lr, dim=0).to(x.device) # b, c, h, w

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x1):
        x2 = self.get_lr_data(x1)
        # encoder1
        res1x_1 = self.conv_input(x1)
        res1x_1 = self.dense0(res1x_1)

        res2x_1 = self.conv2x(res1x_1)
        res2x_1 = self.dense1(res2x_1)

        res4x_1 = self.conv4x(res2x_1)
        res4x_1 = self.dense2(res4x_1)

        res8x_1 = self.conv8x(res4x_1)
        res8x_1 = self.dense3(res8x_1)

        res16x_1 = self.conv16x(res8x_1)
        res16x_1 = self.dense4(res16x_1)

        # encoder2
        res1x_2 = self.conv_input(x2)
        res1x_2 = self.dense0(res1x_2)

        res2x_2 = self.conv2x(res1x_2)
        res2x_2 = self.dense1(res2x_2)

        res4x_2 = self.conv4x(res2x_2)
        res4x_2 = self.dense2(res4x_2)

        res8x_2 = self.conv8x(res4x_2)
        res8x_2 = self.dense3(res8x_2)

        res16x_2 = self.conv16x(res8x_2)
        res16x_2 = self.dense4(res16x_2)

        # dual-perspective feature fusion
        res1x, res2x, res4x, res8x, res16x = self.dpffm(
            [res1x_1, res2x_1, res4x_1, res8x_1, res16x_1],
            [res1x_2, res2x_2, res4x_2, res8x_2, res16x_2]
        )

        # decoder
        res8x1 = self.convd16x(res16x)
        res8x1 = F.interpolate(res8x1, res8x.size()[2:], mode='bilinear')
        res8x2 = self.fusion4(res8x, [res1x, res2x, res4x])
        res8x2 = torch.cat([res8x1, res8x2], dim=1)
        res8x2 = self.dense_4(res8x2)
        res8x2 = self.add_block4(res8x1, res8x, res8x2)

        res4x1 = self.convd8x(res8x2)
        res4x1 = F.interpolate(res4x1, res4x.size()[2:], mode='bilinear')
        res4x2 = self.fusion3(res4x, [res1x, res2x])
        res4x2 = torch.cat([res4x1, res4x2], dim=1)
        res4x2 = self.dense_3(res4x2)
        res4x2 = self.add_block3(res4x1, res4x, res4x2)

        res2x1 = self.convd4x(res4x2)
        res2x1 = F.interpolate(res2x1, res2x.size()[2:], mode='bilinear')
        res2x2 = self.fusion2(res2x, [res1x])
        res2x2 = torch.cat([res2x1, res2x2], dim=1)
        res2x2 = self.dense_2(res2x2)
        res2x2 = self.add_block2(res2x1, res2x, res2x2)

        res1x1 = self.convd2x(res2x2)
        res1x1 = F.interpolate(res1x1, res1x.size()[2:], mode='bilinear')
        res1x2 = torch.cat([res1x1, res1x], dim=1)
        res1x2 = self.dense_1(res1x2)
        res1x2 = self.add_block1(res1x1, res1x, res1x2)

        out = self.head(res1x2)
        out = F.interpolate(out, x1.size()[2:], mode='bilinear')

        return out


if __name__ == "__main__":
    num_classes = 2
    model = MCDNet()
    # inp = torch.randn(size=(2, 3, 256, 256))
    # assert model(input).shape == (2, 2, 256, 256)