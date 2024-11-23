# -*- coding: utf-8 -*-
# @Time    : 2024/7/26 上午11:19
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : dbnet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# from models.Transformer.ViT import truncated_normal_

# Decoder细化卷积模块
class SBR(nn.Module):
    def __init__(self, in_ch):
        super(SBR, self).__init__()
        self.conv1x3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True)
        )
        self.conv3x1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv3x1(self.conv1x3(x))  # 先进行1x3的卷积，得到结果并将结果再进行3x1的卷积
        return out + x


# 下采样卷积模块 stage 1,2,3
class c_stage123(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.stage123 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
        )
        self.conv1x1_123 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        stage123 = self.stage123(x)  # 3*3卷积，两倍下采样    3*224*224-->64*112*112
        max = self.maxpool(x)  # 最大值池化，两倍下采样   3*224*224-->3*112*112
        max = self.conv1x1_123(max)  # 1*1卷积     3*112*112-->64*112*112
        stage123 = stage123 + max  # 残差结构，广播机制
        return stage123


# 下采样卷积模块 stage4,5
class c_stage45(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.stage45 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(),
        )
        self.conv1x1_45 = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        stage45 = self.stage45(x)  # 3*3卷积模块 2倍下采样
        max = self.maxpool(x)  # 最大值池化，两倍下采样
        max = self.conv1x1_45(max)  # 1*1卷积模块 调整通道数
        stage45 = stage45 + max  # 残差结构
        return stage45


class Identity(nn.Module):  # 恒等映射
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# 轻量卷积模块
class DepthwiseConv2d(nn.Module):  # 用于自注意力机制
    def __init__(self, in_chans, out_chans, kernel_size=1, stride=1, padding=0, dilation=1):
        super().__init__()
        # depthwise conv
        self.depthwise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,  # 深层卷积的膨胀率
            groups=in_chans  # 指定分组卷积的组数
        )
        # batch norm
        self.bn = nn.BatchNorm2d(num_features=in_chans)

        # pointwise conv   逐点卷积
        self.pointwise = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_chans,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# residual skip connection 残差跳跃连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, input, **kwargs):
        x = self.fn(input, **kwargs)
        return (x + input)


# layer norm plus 层归一化
class PreNorm(nn.Module):  # 代表神经网络层
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, input, **kwargs):
        return self.fn(self.norm(input), **kwargs)


# FeedForward层使得representation的表达能力更强
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return self.net(input)


class ConvAttnetion(nn.Module):
    '''
    using the Depth_Separable_Wise Conv2d to produce the q, k, v instead of using Linear Project in ViT
    '''

    def __init__(self, dim, img_size, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1,
                 dropout=0., last_stage=False):
        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** (-0.5)

        pad = (kernel_size - q_stride) // 2

        self.to_q = DepthwiseConv2d(in_chans=dim, out_chans=inner_dim, kernel_size=kernel_size, stride=q_stride,
                                    padding=pad)  # 自注意力机制
        self.to_k = DepthwiseConv2d(in_chans=dim, out_chans=inner_dim, kernel_size=kernel_size, stride=k_stride,
                                    padding=pad)
        self.to_v = DepthwiseConv2d(in_chans=dim, out_chans=inner_dim, kernel_size=kernel_size, stride=v_stride,
                                    padding=pad)

        self.to_out = nn.Sequential(
            nn.Linear(
                in_features=inner_dim,
                out_features=dim
            ),
            nn.Dropout(dropout)
        ) if project_out else Identity()

    def forward(self, x):
        b, n, c, h = *x.shape, self.heads  # * 星号的作用大概是去掉 tuple 属性吧

        # print(x.shape)
        # print('+++++++++++++++++++++++++++++++++')

        # if语句内容没有使用
        if self.last_stage:
            cls_token = x[:, 0]
            # print(cls_token.shape)
            # print('+++++++++++++++++++++++++++++++++')
            x = x[:, 1:]  # 去掉每个数组的第一个元素

            cls_token = rearrange(torch.unsqueeze(cls_token, dim=1), 'b n (h d) -> b h n d', h=h)

        # rearrange:用于对张量的维度进行重新变换排序，可用于替换pytorch中的reshape，view，transpose和permute等操作
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)  # [1, 3136, 64]-->1*64*56*56
        # batch_size,N(通道数),h,w

        q = self.to_q(x)  # 1*64*56*56-->1*64*56*56
        # print(q.shape)
        # print('++++++++++++++')
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)  # 1*64*56*56-->1*1*3136*64
        # print(q.shape)
        # print('=====================')
        # batch_size,head,h*w,dim_head

        k = self.to_k(x)  # 操作和q一样
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        # batch_size,head,h*w,dim_head

        v = self.to_v(x)  ##操作和q一样
        # print(v.shape)
        # print('[[[[[[[[[[[[[[[[[[[[[[[[[[[[')
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        # print(v.shape)
        # print(']]]]]]]]]]]]]]]]]]]]]]]]]]]')
        # batch_size,head,h*w,dim_head

        if self.last_stage:
            # print(q.shape)
            # print('================')
            q = torch.cat([cls_token, q], dim=2)
            # print(q.shape)
            # print('++++++++++++++++++')
            v = torch.cat([cls_token, v], dim=2)
            k = torch.cat([cls_token, k], dim=2)

        # calculate attention by matmul + scale
        # permute:(batch_size,head,dim_head,h*w
        # print(k.shape)
        # print('++++++++++++++++++++')
        k = k.permute(0, 1, 3, 2)  # 1*1*3136*64-->1*1*64*3136
        # print(k.shape)
        # print('====================')
        attention = (q.matmul(k))  # 1*1*3136*3136
        # print(attention.shape)
        # print('--------------------')
        attention = attention * self.scale  # 可以得到一个logit的向量，避免出现梯度下降和梯度爆炸
        # print(attention.shape)
        # print('####################')
        # pass a softmax
        attention = F.softmax(attention, dim=-1)
        # print(attention.shape)
        # print('********************')

        # matmul v
        # attention.matmul(v):(batch_size,head,h*w,dim_head)
        # permute:(batch_size,h*w,head,dim_head)
        out = (attention.matmul(v)).permute(0, 2, 1, 3).reshape(b, n,
                                                                c)  # 1*3136*64  这些操作的目的是将注意力权重和值向量相乘后得到的结果进行重塑，得到一个形状为 (batch size, 序列长度, 值向量或矩阵的维度) 的张量

        # linear project
        out = self.to_out(out)
        return out


# Reshape Layers
class Rearrange(nn.Module):
    def __init__(self, string, h, w):
        super().__init__()
        self.string = string
        self.h = h
        self.w = w

    def forward(self, input):

        if self.string == 'b c h w -> b (h w) c':
            N, C, H, W = input.shape
            # print(input.shape)
            x = torch.reshape(input, shape=(N, -1, self.h * self.w)).permute(0, 2, 1)
            # print(x.shape)
            # print('+++++++++++++++++++')
        if self.string == 'b (h w) c -> b c h w':
            N, _, C = input.shape
            # print(input.shape)
            x = torch.reshape(input, shape=(N, self.h, self.w, -1)).permute(0, 3, 1, 2)
            # print(x.shape)
            # print('=====================')
        return x


# Transformer layers
class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([  # 管理子模块，参数注册
            nn.ModuleList([
                PreNorm(dim=dim, fn=ConvAttnetion(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout,
                                                  last_stage=last_stage)),  # 归一化，重参数化
                PreNorm(dim=dim, fn=FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x


class DBNet(nn.Module):  # 最主要的大函数
    def __init__(self, img_size, in_channels, num_classes, dim=64, kernels=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                 heads=[1, 3, 6, 6],
                 depth=[1, 2, 10, 10], pool='cls', dropout=0., emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in ['cls', 'mean'], f'pool type must be either cls or mean pooling'
        self.pool = pool
        self.dim = dim

        # stage1
        # k:7 s:4    in: 1, 64, 56, 56  out: 1, 3136, 64
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(  # 1*3*224*224-->[1, 64, 56, 56]
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[0],
                stride=strides[0],
                padding=2
            ),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),  # [1, 64, 56, 56]-->[1, 3136, 64]
            nn.LayerNorm(dim)  # 对每个batch归一化
        )

        self.stage1_transformer = nn.Sequential(
            Transformer(  #
                dim=dim,
                img_size=img_size // 4,
                depth=depth[0],  # Transformer层中的编码器和解码器层数。
                heads=heads[0],
                dim_head=self.dim,  # 它是每个注意力头的维度大小，通常是嵌入维度除以头数。
                mlp_dim=dim * scale_dim,  # mlp_dim:它是Transformer中前馈神经网络的隐藏层维度大小，通常是嵌入维度乘以一个缩放因子。
                dropout=dropout,
                # last_stage=last_stage     #它是一个标志位，用于表示该Transformer层是否是最后一层。
            ),
            Rearrange('b (h w) c -> b c h w', h=img_size // 4, w=img_size // 4)
        )

        # stage2
        # k:3 s:2  in: 1, 192, 28, 28  out: 1, 784, 192
        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim

        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[1],
                stride=strides[1],
                padding=1
            ),
            Rearrange('b c h w -> b (h w) c', h=img_size // 8, w=img_size // 8),
            nn.LayerNorm(dim)
        )

        self.stage2_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=img_size // 8,
                depth=depth[1],
                heads=heads[1],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout
            ),
            Rearrange('b (h w) c -> b c h w', h=img_size // 8, w=img_size // 8)
        )

        # stage3
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim

        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[2],
                stride=strides[2],
                padding=1
            ),
            Rearrange('b c h w -> b (h w) c', h=img_size // 16, w=img_size // 16),
            nn.LayerNorm(dim)
        )

        self.stage3_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=img_size // 16,
                depth=depth[2],
                heads=heads[2],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout
            ),
            Rearrange('b (h w) c -> b c h w', h=img_size // 16, w=img_size // 16)
        )

        # stage4
        in_channels = dim
        scale = heads[3] // heads[2]
        dim = scale * dim

        self.stage4_conv_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernels[3],
                stride=strides[3],
                padding=1
            ),
            Rearrange('b c h w -> b (h w) c', h=img_size // 32, w=img_size // 32),
            nn.LayerNorm(dim)
        )

        self.stage4_transformer = nn.Sequential(
            Transformer(
                dim=dim, img_size=img_size // 32,
                depth=depth[3],
                heads=heads[3],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout,
            ),
            Rearrange('b (h w) c -> b c h w', h=img_size // 32, w=img_size // 32)
        )

        ### CNN Branch ###
        self.c_stage1 = c_stage123(in_chans=3, out_chans=64)
        self.c_stage2 = c_stage123(in_chans=64, out_chans=128)
        self.c_stage3 = c_stage123(in_chans=128, out_chans=384)
        self.c_stage4 = c_stage45(in_chans=384, out_chans=512)
        self.c_stage5 = c_stage45(in_chans=512, out_chans=1024)
        self.c_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up_conv1 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.up_conv2 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=1)

        ### CTmerge ###
        self.CTmerge1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.CTmerge2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.CTmerge3 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )

        self.CTmerge4 = nn.Sequential(
            nn.Conv2d(in_channels=896, out_channels=640, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(in_channels=640, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # decoder
        self.decoder4 = nn.Sequential(
            DepthwiseConv2d(
                in_chans=1408,
                out_chans=1024,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            DepthwiseConv2d(
                in_chans=1024,
                out_chans=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU()
        )
        self.decoder3 = nn.Sequential(
            DepthwiseConv2d(
                in_chans=896,
                out_chans=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            DepthwiseConv2d(
                in_chans=512,
                out_chans=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU()
        )

        self.decoder2 = nn.Sequential(
            DepthwiseConv2d(
                in_chans=576,
                out_chans=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            DepthwiseConv2d(
                in_chans=256,
                out_chans=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU()
        )

        self.decoder1 = nn.Sequential(
            DepthwiseConv2d(
                in_chans=256,
                out_chans=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            DepthwiseConv2d(
                in_chans=64,
                out_chans=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU()
        )
        self.sbr4 = SBR(512)
        self.sbr3 = SBR(384)
        self.sbr2 = SBR(192)
        self.sbr1 = SBR(16)

        self.head = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        ### encoder ###
        # stage1 = ts1  cat  cs1
        # t_s1 = self.t_stage1(input)
        # print(input.shape)
        # print('++++++++++++++++++++++')

        t_s1 = self.stage1_conv_embed(input)  # 1*3*224*224-->1*3136*64

        # print(t_s1.shape)
        # print('======================')

        t_s1 = self.stage1_transformer(t_s1)  # 1*3136*64-->1*64*56*56

        # print(t_s1.shape)
        # print('----------------------')

        c_s1 = self.c_stage1(input)  # 1*3*224*224-->1*64*112*112

        # print(c_s1.shape)
        # print('!!!!!!!!!!!!!!!!!!!!!!!')

        stage1 = self.CTmerge1(torch.cat([t_s1, self.c_max(c_s1)], dim=1))  # 1*64*56*56   # 拼接两条分支

        # print(stage1.shape)
        # print('[[[[[[[[[[[[[[[[[[[[[[[')

        # stage2 = ts2 up cs2
        # t_s2 = self.t_stage2(stage1)
        t_s2 = self.stage2_conv_embed(stage1)  # 1*64*56*56-->1*784*192   # stage2_conv_embed是转化为序列操作

        # print(t_s2.shape)
        # print('[[[[[[[[[[[[[[[[[[[[[[[')
        t_s2 = self.stage2_transformer(t_s2)  # 1*784*192-->1*192*28*28
        # print(t_s2.shape)
        # print('+++++++++++++++++++++++++')

        c_s2 = self.c_stage2(c_s1)  # 1*64*112*112-->1*128*56*56
        stage2 = self.CTmerge2(
            torch.cat([c_s2, F.interpolate(t_s2, size=c_s2.size()[2:], mode='bilinear', align_corners=True)],
                      dim=1))  # mode='bilinear'表示使用双线性插值  1*128*56*56

        # stage3 = ts3 cat cs3
        # t_s3 = self.t_stage3(t_s2)
        t_s3 = self.stage3_conv_embed(t_s2)  # 1*192*28*28-->1*196*384
        # print(t_s3.shape)
        # print('///////////////////////')
        t_s3 = self.stage3_transformer(t_s3)  # 1*196*384-->1*384*14*14
        # print(t_s3.shape)
        # print('....................')
        c_s3 = self.c_stage3(stage2)  # 1*128*56*56-->1*384*28*28
        stage3 = self.CTmerge3(torch.cat([t_s3, self.c_max(c_s3)], dim=1))  # 1*384*14*14

        # stage4 = ts4 up cs4
        # t_s4 = self.t_stage4(stage3)
        t_s4 = self.stage4_conv_embed(stage3)  # 1*384*14*14-->1*49*384
        # print(t_s4.shape)
        # print(';;;;;;;;;;;;;;;;;;;;;;;')
        t_s4 = self.stage4_transformer(t_s4)  # 1*49*384-->1*384*7*7
        # print(t_s4.shape)
        # print('::::::::::::::::::::')

        c_s4 = self.c_stage4(c_s3)  # 1*384*28*28-->1*512*14*14
        stage4 = self.CTmerge4(
            torch.cat([c_s4, F.interpolate(t_s4, size=c_s4.size()[2:], mode='bilinear', align_corners=True)],
                      dim=1))  # 1*512*14*14

        # cs5
        c_s5 = self.c_stage5(stage4)  # 1*512*14*14-->1*1024*7*7

        ### decoder ###
        decoder4 = torch.cat([c_s5, t_s4], dim=1)  # 1*1408*7*7
        decoder4 = self.decoder4(decoder4)  # 1*1408*7*7-->1*512*7*7
        decoder4 = F.interpolate(decoder4, size=c_s3.size()[2:], mode='bilinear',
                                 align_corners=True)  # 1*512*7*7-->1*512*28*28
        decoder4 = self.sbr4(decoder4)  # 1*512*28*28
        # print(decoder4.shape)

        decoder3 = torch.cat([decoder4, c_s3], dim=1)  # 1*896*28*28
        decoder3 = self.decoder3(decoder3)  # 1*384*28*28
        decoder3 = F.interpolate(decoder3, size=t_s2.size()[2:], mode='bilinear', align_corners=True)  # 1*384*28*28
        decoder3 = self.sbr3(decoder3)  # 1*384*28*28
        # print(decoder3.shape)

        decoder2 = torch.cat([decoder3, t_s2], dim=1)  # 1*576*28*28
        decoder2 = self.decoder2(decoder2)  # 1*192*28*28
        decoder2 = F.interpolate(decoder2, size=c_s1.size()[2:], mode='bilinear', align_corners=True)  # 1*192*112*112
        decoder2 = self.sbr2(decoder2)  # 1*192*112*112
        # print(decoder2.shape)

        decoder1 = torch.cat([decoder2, c_s1], dim=1)  # 1*256*112*112
        decoder1 = self.decoder1(decoder1)  # 1*16*112*112
        # print(decoder1.shape)
        final = F.interpolate(decoder1, size=input.size()[2:], mode='bilinear', align_corners=True)  # 1*16*224*224
        # print(final.shape)
        # final = self.sbr1(decoder1)
        # print(final.shape)
        final = self.head(final)  # 1*3*224*224

        return final


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224).cuda()
    model = DBNet(img_size=224, in_channels=3, num_classes=7).cuda()
    y = model(x)
    print(y.shape)
    # torch.Size([1, 7, 224, 224])