from typing import Optional, Callable, Union, Tuple

import torch
import torch.nn as nn

from .utils import get_same_padding_conv2d, get_same_padding_maxPool2d
from .squeeze_excitation import SqueezeExcitation


class BasicUpBlock(nn.Module):
    """
    Original paper:
    https://arxiv.org/pdf/1603.05027.pdf
    Inspiration from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int]] = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 ratio: float = 1./16,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_func: Optional[Callable[..., nn.Module]] = None,
                 ):
        super().__init__()

        stride = stride if isinstance(stride, (list, tuple)) else (stride, stride)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.upsample layers upsample the input when stride != 1

        Conv2d = get_same_padding_conv2d(image_size=None)
        self.norm0 = norm_layer(in_channels)
        self.conv0 = Conv2d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            dilation=dilation,
                            groups=groups,
                            )

        self.norm1 = norm_layer(in_channels)
        self.conv1 = Conv2d(in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            bias=bias,
                            dilation=dilation,
                            groups=groups,
                            )

        self.norm2 = norm_layer(in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=stride,
                                     stride=stride,
                                     bias=bias,
                                     dilation=dilation,
                                     groups=groups,
                                     )


        if activation_func is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation_func

        self.se = SqueezeExcitation(channels=out_channels, ratio=ratio)

        self.stride = stride

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

        # Transpose convolution
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)


        return x
