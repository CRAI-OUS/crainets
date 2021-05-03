from typing import Optional, Callable, Union, Tuple

import torch
import torch.nn as nn

from ..blocks.utils import get_same_padding_conv2d, get_same_padding_maxPool2d
from .squeeze_excitation import SqueezeExcitation

class Bottleneck(nn.Module):
    """
    Original paper:
    https://arxiv.org/pdf/1603.05027.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self,
                 channels: int,
                 mid_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 ratio: float = 1./16,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_func: Optional[Callable[..., nn.Module]] = None,
                 ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)

        Conv2d = get_same_padding_conv2d(image_size=None)
        self.norm0 = norm_layer(channels)

        self.conv1 = Conv2d(in_channels=channels,
                            out_channels=mid_channels,
                            kernel_size=1,
                            stride=1,
                            bias=bias,
                            )
        self.norm1 = norm_layer(mid_channels)

        self.conv2 = Conv2d(in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            stride=stride,
                            groups=groups,
                            dilation=dilation,
                            bias=bias,
                            )
        self.norm2 = norm_layer(mid_channels)

        self.conv3 = Conv2d(in_channels=mid_channels,
                            out_channels=channels,
                            kernel_size=1)

        if activation_func is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation_func

        self.se = SqueezeExcitation(channels=channels, ratio=ratio)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.norm0(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)

        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv3(x)

        x = self.se(x)
        x += identity

        return x
