#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:08:10 2021

@author: lidia
"""
"""Adapted from https://github.com/jphdotam/Unet3D"""
import torch.nn as nn


class 3Dcnn(nn.Module):
    def __init__(self, n_channels, width_multiplier=1, use_ds_conv=False):
        """A simple 3D CNN + fully connected layer for classification. Adapted 
        from a 3D Unet from https://github.com/jphdotam/Unet3D
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 
                  higher values increase the number of kernels pay layer, by that factor
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d
        self.in_fc = None # Is it possible to get from x? Otherwise must be argument?
        self.out_fc = self.channels[4]

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down4 = Down(self.channels[3], self.channels[4], conv_type=self.convtype)
        self.down5 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = FullyConnected(self.in_fc, self.out_fc)
    
    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = x.flatten()    
        out = self.fc(x)

        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class FullyConnected(nn.Module):
    """Two fully connected layers"""
    def __init__(self, in_features, out_features, dropout=0.25, mid_features=None):
        super().__init__()
        if not mid_features:
            mid_features = in_features
            
        self.two_layers = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.BatchNorm1d(mid_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(mid_features, out_features)
            )
            
    def forward(self, x):
        return self.two_layers(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super().__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, 
                                   padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out