"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""
from typing import Union, Tuple
from torch import nn

from deepness.models.efficientnet.blocks.squeeze_excitation import SqueezeExcitation

from deepness.models.efficientnet.blocks.swish import (
    Swish,
    MemoryEfficientSwish,
    )

from deepness.models.efficientnet.blocks.utils.conv_pad import get_same_padding_conv2d
from deepness.models.efficientnet.blocks.utils.utils import (
    drop_connect,
    calculate_output_image_size,
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]],
                 se_ratio: float = 0.25,
                 expand_ratio: Union[int, float] = 1.,
                 id_skip: bool = True,
                 norm_method: Union[str] = 'batch_norm',
                 batch_norm_momentum: float = 0.99,
                 batch_norm_epsilon: float = 0.001,
                 image_size: Union[int, Tuple[int]] = None,
                 ):
        super().__init__()
        self.batch_norm_momentum = 1 - batch_norm_momentum # pytorch's difference from tensorflow
        self.batch_norm_epsilon = batch_norm_epsilon
        self.norm_method = norm_method

        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.id_skip = id_skip  # whether to use skip connection and drop connect
        self.expand_ratio = expand_ratio
        self.stride = stride

        # Expansion phase (Inverted Bottleneck)
        self.in_channels = in_channels
        self.mid_channels = int(in_channels * self.expand_ratio)  # number of mid channels
        self.out_channels = out_channels

        if self.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self.expand_conv = Conv2d(in_channels=in_channels,
                                      out_channels=self.mid_channels,
                                      kernel_size=1,
                                      bias=False,
                                      )
            self.norm0 = self._norm_method(output=self.mid_channels)

        # Depthwise convolution phase
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.depthwise_conv = Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            groups=self.mid_channels,  # groups makes it depthwise
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            )

        self.norm1 = self._norm_method(output=self.mid_channels)

        image_size = calculate_output_image_size(image_size, stride)
        # Squeeze and Excitation layer, if desired
        if self.has_se:
            # ratio such that the number of squeezed channels are ratio(0.25)*in_channels from efficientnet architecture
            self.squeeze_excite = SqueezeExcitation(channels=self.mid_channels,
                                                    ratio=se_ratio*self.in_channels/self.mid_channels)

        # Pointwise convolution phase
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.project_conv = Conv2d(in_channels=self.mid_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   bias=False,
                                   )
        self.norm2 = self._norm_method(output=out_channels)
        self.swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self.expand_conv(inputs)
            x = self.norm0(x)
            x = self.swish(x)

        x = self.depthwise_conv(x)
        x = self.norm1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x = self.squeeze_excite(x)

        # Pointwise Convolution
        x = self.project_conv(x)
        x = self.norm2(x)

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()

    def _norm_method(self, output: int):
        norms = {
            'batch_norm': nn.BatchNorm2d(
                num_features=output,
                momentum=self.batch_norm_momentum,
                eps=self.batch_norm_epsilon,
                ),
            'instance_norm': nn.InstanceNorm2d(
                num_features=output,
                momentum=self.batch_norm_momentum,
                eps=self.batch_norm_epsilon,
                ),
            'layer_norm': nn.LayerNorm(
                normalized_shape=output,
                eps=self.batch_norm_epsilon,
                ),
            }
        return norms[self.norm_method]
