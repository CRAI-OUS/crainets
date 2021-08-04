import math
from functools import partial
from typing import Union, Tuple

from torch import nn
from torch.nn import functional as F

# Note:
# The following 'SamePadding' functions make output size equal ceil(input size*stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names!!!


def get_same_padding_maxPool2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]],
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False,
                 ):
        super().__init__(kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         return_indices=return_indices,
                         ceil_mode=ceil_mode,
                         )
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])

        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]],
                 padding: Union[int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False,
                 image_size: Union[int, Tuple[int], None] = None,
                 ):
        super().__init__(kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         return_indices=return_indices,
                         ceil_mode=ceil_mode,
                         )
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)
        return x
