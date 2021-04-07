# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# and
# Stacked Hourglass https://github.com/princeton-vl/pytorch_stacked_hourglass
# Copyright (c) 2019, princeton-vl
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn = self.bn(linear) if self.with_bn else linear
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv1 = convolution(k, inp_dim, out_dim, stride=stride, with_bn=with_bn)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        # self.conv2 = convolution(k, out_dim, out_dim, stride=stride, with_bn=with_bn)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


class Hourglass(nn.Module):
    def __init__(self, dims, n=0, in_channels=3):
        super(Hourglass, self).__init__()
        curr_dim = in_channels
        self.pool1 = nn.MaxPool2d(2, 2)
        self.up1 = residual(curr_dim, curr_dim)  # keep channel size
        self.low1 = residual(curr_dim, dims[n])  # change channel size

        # recursive
        if n < len(dims) - 1:
            self.low2 = Hourglass(dims, n + 1, in_channels=dims[n])
        else:
            self.low2 = residual(dims[n], dims[n])
        self.low3 = residual(dims[n], curr_dim)  # back to original channel size
        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # x.shape = (H, W, C1)
        up1 = self.up1(x)   # (H, W, C1)
        x = self.pool1(x)   # (H/2, W/2, C1)
        x = self.low1(x)    # (H/2, W/2, C2)
        x = self.low2(x)    # (H/2, W/2, C2)
        x = self.low3(x)    # (H/2, W/2, C1)
        up2 = self.up2(x)   # (H, W, C1)
        return up1 + up2    # (H, W, C1)


def make_pre_layer(in_channels, pre_dims=(128, 256)):
    layer = nn.Sequential(convolution(7, in_channels, pre_dims[0], stride=2),
                          residual(pre_dims[0], pre_dims[1], stride=2))
    downsampling_factor = 4
    out_dim = pre_dims[-1]
    return layer, downsampling_factor, out_dim


class StackedHourglass(nn.Module):
    def __init__(self, in_channels, dims, num_classes):
        super(StackedHourglass, self).__init__()
        self.num_classes = num_classes

        self.pre, self.downsampling_factor, pre_out_dim = make_pre_layer(in_channels)
        self.hg1 = Hourglass(dims[0], in_channels=pre_out_dim)
        self.hg2 = Hourglass(dims[1], in_channels=dims[0][-1])
        self.inter_layer = residual(dims[0][0], dims[1][0])

        # networks prediction
        self.hm = nn.Sequential(nn.Conv2d(dims[1][0], num_classes, kernel_size=3, padding=1, stride=1),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        x = self.pre(x)
        x = self.hg1(x)
        x = self.inter_layer(x)
        x = self.hg2(x)
        out = self.hm(x)
        return out


