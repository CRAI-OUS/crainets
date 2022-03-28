"""
Copyright (c) 2021, CRAI
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Union, Tuple
import math
from functools import partial

from torch import nn
from torch.nn import functional as F

# NOTE!
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names!!!

def get_same_padding_conv3d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv3dDynamicSamePadding
    else:
        return partial(Conv3dStaticSamePadding, image_size=image_size)


class Conv3dDynamicSamePadding(nn.Conv3d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """
    # NOTE! put this in the docstring as an example?
    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Union[int, Tuple[int]] = 1,
                 bias: bool = True,
                 ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=0,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         )
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3

    def forward(self, x):
        ih, iw, ide = x.size()[-3:]
        kh, kw, kde = self.weight.size()[-3:]
        sh, sw, sde = self.stride
        oh, ow, ode = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(ide / sde) # change the output size according to stride!!!
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_d = max((ode - 1) * self.stride[2] + (kde - 1) * self.dilation[2] + 1 - ide, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2,
                          pad_d // 2, pad_d - pad_d // 2])
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv3dStaticSamePadding(nn.Conv3d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 dialiation: Union[int, Tuple[int]] = 1,
                 groups: Union[int, Tuple[int]] = 1,
                 bias: bool = True,
                 image_size: Union[int, Tuple[int], None] = None,
                 ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=0,
                         dilation=dialiation,
                         groups=groups,
                         bias=bias,
                         )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw, ide = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw, kde = self.weight.size()[-3:]  # Extract kernel size
        sh, sw, sde = self.stride
        oh, ow, ode = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(ide / sde)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_d = max((ode - 1) * self.stride[2] + (kde - 1) * self.dilation[2] + 1 - ide, 0)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            self.static_padding = nn.ConstantPad3d(
                (pad_w // 2, pad_w - pad_w // 2,
                 pad_h // 2, pad_h - pad_h // 2,
                 pad_d // 2, pad_d - pad_d // 2),
                value=0)
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
