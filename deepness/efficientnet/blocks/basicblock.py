from typing import Optional, Callable, Union, Tuple

import torch
import torch.nn as nn

from .utils import Conv2dDynamicSamePadding
from .squeeze_excitation import SqueezeExcitation

class BasicBlock(nn.Module):
    """
    Original paper:
    https://arxiv.org/pdf/1603.05027.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self,
                 channels: int,
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

        Conv2d = Conv2dDynamicSamePadding

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.norm0 = norm_layer(channels)
        self.conv0 = Conv2d(in_channels=channels,
                            out_channels=channels,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            dilation=dilation,
                            groups=groups,
                            )
        self.norm1 = norm_layer(channels)


        self.conv1 = Conv2d(in_channels=channels,
                            out_channels=channels,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            dilation=dilation,
                            groups=groups,
                            )

        if activation_func is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation_func

        self.se = SqueezeExcitation(channels=channels, ratio=ratio)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x
        x = self.norm0(x)
        x = self.activation(x)
        x = self.conv0(x)

        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.se(x)
        x += identity

        return x
