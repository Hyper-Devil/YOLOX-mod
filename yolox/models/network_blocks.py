#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .hornet import Block, gnconv, LayerNorm
from timm.models.layers import DropPath
import numpy as np

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Bconv(nn.Module):
    def __init__(self,ch_in,ch_out,k,s):
        '''
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param k: 卷积核尺寸
        :param s: 步长
        :return:
        '''
        super(Bconv, self).__init__()
        self.conv=nn.Conv2d(ch_in,ch_out,k,s,padding=k//2)
        self.bn=nn.BatchNorm2d(ch_out)
        self.act=nn.SiLU()
    def forward(self,x):
        '''
        :param x: 输入
        :return:
        '''
        return self.act(self.bn(self.conv(x)))

class MPConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        '''
        :param ch_in: 输如通道
        :param ch_out: 这里给的是中间层的输出通道
        '''
        super(MPConv, self).__init__()
        #分支一
        self.conv1=nn.Sequential(
            nn.MaxPool2d(2,2),
            Bconv(ch_in,ch_out,1,1),
        )
        #分支二
        self.conv2=nn.Sequential(
            Bconv(ch_in,ch_out,1,1),
            Bconv(ch_out,ch_out,3,2),
        )

    def forward(self,x):
        #分支一输出
        output1=self.conv1(x)

        #分支二输出
        output2=self.conv2(x)
        return torch.cat((output1,output2),dim=1)

class E_ELAN(nn.Module):
    def __init__(self,ch_in,ch_out,flg=False):
        '''
        :param ch_in: 输入通道
        :param ch_out: 这里给的是中间层的输出通道
        :param flg: 判断是否为backbone的最后一层，因为这里的输出通道数有所改变
        '''
        super(E_ELAN, self).__init__()
        # 卷积类型一
        self.conv1=Bconv(ch_in,ch_out,k=1,s=1)
        # 卷积类型二
        self.conv2=Bconv(ch_out,ch_out,k=3,s=1)

        #cat之后的卷积
        if flg:
            self.conv3=Bconv(2*ch_in,ch_in,k=1,s=1)
        else:
            self.conv3=Bconv(2*ch_in,2*ch_in,k=1,s=1)

    def forward(self,x):
        '''
        :param x: 输入
        :return:
        '''
        #分支一输出
        output1=self.conv1(x)

        #分支二输出
        output2_1=self.conv1(x)
        output2_2=self.conv2(output2_1)
        output2_3=self.conv2(output2_2)
        output2_4=self.conv2(output2_3)
        output2_5=self.conv2(output2_4)
        output_cat=torch.cat((output1, output2_1, output2_3, output2_5), dim=1)
        return self.conv3(output_cat)
        
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


