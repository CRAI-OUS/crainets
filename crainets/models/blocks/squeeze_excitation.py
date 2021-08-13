"""
Copyright (c) 2021, CRAI
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    """
    Waiting for proper docstring...
    """

    def __init__(self,
                 channels: int,
                 ratio: float = 1./16,
                 ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeezed_channels = max(1, int(channels*ratio))
        self.layer_1 = nn.Conv2d(in_channels=channels,
                                 out_channels=squeezed_channels,
                                 kernel_size=1,
                                 bias=True,
                                 )
        self.layer_2 = nn.Conv2d(in_channels=squeezed_channels,
                                 out_channels=channels,
                                 kernel_size=1,
                                 bias=True)
        self.silu = torch.nn.SiLU()
        # Could do this using linear layer aswell, but than we need to .view in forward
        # self.linear_1 = nn.Linear(in_features=channels, out_features=squeezed_channels, bias=True)
        # self.linear_2 = nn.Linear(in_features=squeezed_channels, out_features=channels, bias=True)

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.layer_1(x)
        x = self.silu(x)
        x = self.layer_2(x)
        x = torch.sigmoid(x) * inputs
        return x
