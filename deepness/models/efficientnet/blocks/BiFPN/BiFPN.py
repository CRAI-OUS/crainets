from typing import List, Optional, Callable, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..swish import MemoryEfficientSwish, Swish
from ..utils import Conv2dDynamicSamePadding


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117 modified by JonOttesen
    Source: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 norm=True,
                 activation=False,
                 onnx_export=False,
                 ):

        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dDynamicSamePadding(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=3,
                                                       stride=1,
                                                       groups=in_channels,
                                                       bias=False,
                                                       )
        self.pointwise_conv = Conv2dDynamicSamePadding(in_channels=out_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=1,
                                                       stride=1,
                                                       bias=True
                                                       )

        self.norm = norm
        if self.norm:
            self.i_norm = nn.InstanceNorm2d(num_features=out_channels)
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            # self.i_norm = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.i_norm(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117,
    Re-implemented by JonOttesen
    """

    def __init__(self,
                 channels: Union[List[int], int],
                 layers: int,
                 epsilon: float = 1e-4,
                 onnx_export: bool = False,
                 attention: bool = True,
                 ):
        """
        Args:
            num_channels:
            conv_channels:
            attention (bool): whether to use attention gates to weight input
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        # Create channels in layer
        if isinstance(channels, int):
            channels = [channels for i in range(layers)]

        self.epsilon = epsilon

        # Conv layers, i.e the middle node, from the image this should be the 3 middle nodes
        # Middle layer
        self.middle_conv = nn.ModuleList([
            SeparableConvBlock(channel, onnx_export=onnx_export) for channel in channels[1:-1]
            ])

        # Last layer, in the image this should be the 5 last nodes
        self.out_conv = nn.ModuleList([
            SeparableConvBlock(in_channels=i, out_channels=i, onnx_export=onnx_export)
            for i in channels
            ])

        ### Important, both of these have the higher spatial dimensions first
        ### The instance norm and activation function are original to me in this architecture

        # The upsampling, only with transpose convolutions to allow different number of channels
        # This is the four arrows going downwards(i.e higher spatial dimension) in the image figure
        self.up_layers = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels[i + 1],
                               out_channels=channels[i],
                               kernel_size=2,
                               stride=2,
                               bias=True),
            nn.InstanceNorm2d(channels[i]),
            MemoryEfficientSwish() if not onnx_export else Swish(),
            )
            for i in range(len(channels) - 1)])

        # The downsampling, only with convolutional layers to allow different number of channels
        # This is the four arrows going upwards in the figure (i.e lower spatial dimension) on the last column

        self.down_layers = nn.ModuleList([nn.Sequential(
            Conv2dDynamicSamePadding(in_channels=channels[i],
                                     out_channels=channels[i + 1],
                                     kernel_size=3,
                                     stride=2,
                                     bias=True),
            nn.InstanceNorm2d(channels[i + 1]),
            MemoryEfficientSwish() if not onnx_export else Swish(),
            )
            for i in range(len(channels) - 1)])
        # """

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Weight
        if attention:
            self.middle_attention = [
                nn.Parameter(torch.ones(2, dtype=torch.float32, requires_grad=True))
                for i in range(len(channels) - 2)]

            # Weights for the last layer
            self.out_attention = [nn.Parameter(torch.ones(2, dtype=torch.float32, requires_grad=True))]
            self.out_attention.extend(
                [nn.Parameter(torch.ones(3, dtype=torch.float32, requires_grad=True))
                for i in range(len(channels) - 2)
                ])
            self.out_attention.append(nn.Parameter(torch.ones(2, dtype=torch.float32, requires_grad=True)))
            self.weight_relu = nn.ReLU()

        self.middle_attention = nn.ParameterList(self.middle_attention)
        self.out_attention = nn.ParameterList(self.out_attention)

        self.attention = attention

    def forward(self, inputs: Union[List[torch.Tensor], Tuple[torch.Tensor]]):
        """
        Input assumes that the input dimension decreases by 2 for each input
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        assert len(inputs) > 2, 'the length of inputs must be larger than 2'

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        middle = list()
        outputs = list()

        # These are the middle nodes
        x = inputs[-1]
        for i in range(1, len(self.middle_conv) + 1):
            weights = self.weight_relu(self.middle_attention[-i])
            weight = weights / (torch.sum(weights, dim=0) + self.epsilon)
            x = self.middle_conv[-i](self.swish(inputs[-i - 1]*weight[0] + self.up_layers[-i](x)*weight[1]))
            middle.append(x)
        middle.reverse()

        # ----------------------------------------------------

        # The top output layer
        weights = self.weight_relu(self.out_attention[0])
        weight = weights / (torch.sum(weights, dim=0) + self.epsilon)
        x = self.out_conv[0](self.swish(inputs[0]*weight[0] + self.up_layers[0](middle[0])*weight[1]))
        outputs.append(x)

        # The middle layer
        for i in range(1, len(self.out_conv) - 1):
            weights = self.weight_relu(self.out_attention[i])
            weight = weights / (torch.sum(weights, dim=0) + self.epsilon)
            x = self.out_conv[i](self.swish(inputs[i]*weight[0] + middle[i-1]*weight[1] + self.down_layers[i-1](x)*weight[2]))
            outputs.append(x)

        # The bottom output layer
        weights = self.weight_relu(self.out_attention[-1])
        weight = weights / (torch.sum(weights, dim=0) + self.epsilon)

        x = self.out_conv[-1](self.swish(inputs[-1]*weight[0] + self.down_layers[-1](x)*weight[1]))
        outputs.append(x)

        return outputs


    def _forward(self, inputs: Union[List[torch.Tensor], Tuple[torch.Tensor]]):

        middle = list()
        outputs = list()

        # These are the middle nodes
        x = inputs[-1]
        for i in range(1, len(self.middle_conv) + 1):
            x = self.middle_conv[-i](self.swish(self.up_layers[-i](x) + inputs[-i - 1]))
            middle.append(x)
        middle.reverse()

        # ----------------------------------------------------

        # The top output layer
        x = self.out_conv[0](self.swish(inputs[0] + self.up_layers[0](middle[0])))
        outputs.append(x)

        # The middle layer
        for i in range(1, len(self.out_conv) - 1):
            x = self.out_conv[i](self.swish(inputs[i] + middle[i-1] + self.down_layers[i-1](x)))
            outputs.append(x)

        # The bottom output layer
        x = self.out_conv[-1](self.swish(inputs[-1] + self.down_layers[-1](x)))
        outputs.append(x)

        return outputs
