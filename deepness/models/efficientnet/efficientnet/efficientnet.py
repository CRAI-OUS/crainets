from typing import Union, List
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..blocks.utils import get_same_padding_conv2d

from .utils import (
    round_filters,
    round_repeats,
    calculate_output_image_size,
)

from ..blocks import (
    MBConvBlock,
    Swish,
    MemoryEfficientSwish,
    )

from .config import (
    BlockArgs,
    GlobalParams,
    VALID_MODELS,
    efficientnet_params,
    blocks,
    GlobalParams
    )


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    Example:


        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self,
                 in_channels: int,
                 blocks_args: List[BlockArgs],
                 global_params: GlobalParams):
        super().__init__()

        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'

        self.global_params = global_params
        self.blocks_args = blocks_args

        # Batch norm parameters
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        stem_stride = 2
        out_channels = round_filters(32, global_params)  # number of output channels
        self.conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=stem_stride, bias=False)
        self.norm0 = self._norm_method(output=out_channels)

        image_size = calculate_output_image_size(image_size, stride=stem_stride)

        # Build blocks
        self.blocks = nn.ModuleList([])
        for block_args in blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args.input_filters = round_filters(block_args.input_filters, global_params)
            block_args.output_filters = round_filters(block_args.output_filters, global_params)
            block_args.num_repeat = round_repeats(block_args.num_repeat, global_params)

            # The first block needs to take care of stride and filter size increase.
            self.blocks.append(MBConvBlock(
                in_channels=block_args.input_filters,
                out_channels=block_args.output_filters,
                kernel_size=block_args.kernel_size,
                stride=block_args.stride,
                se_ratio=block_args.se_ratio,
                expand_ratio=block_args.expand_ratio,
                id_skip=block_args.id_skip,
                norm_method=global_params.norm_method,
                batch_norm_momentum=global_params.batch_norm_momentum,
                batch_norm_epsilon=global_params.batch_norm_epsilon,
                ))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args.input_filters = block_args.output_filters
                block_args.stride = 1

            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(MBConvBlock(
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    se_ratio=block_args.se_ratio,
                    expand_ratio=block_args.expand_ratio,
                    id_skip=block_args.id_skip,
                    norm_method=global_params.norm_method,
                    batch_norm_momentum=global_params.batch_norm_momentum,
                    batch_norm_epsilon=global_params.batch_norm_epsilon,
                    ))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self.conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = self._norm_method(output=out_channels)

        self.swish = MemoryEfficientSwish()

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        # Stem
        x = self.swish(self.norm0(self.conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self.swish(self.norm1(self.conv_head(x)))
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self.swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self.blocks:
            block.set_swish(memory_efficient)

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
        return norms[self.global_params.norm_method]

    @classmethod
    def from_name(cls, model_name: str, in_channels: int = 3):
        """
        """
        assert model_name in VALID_MODELS, 'the model: {} not found'.format(model_name)

        args_dict = efficientnet_params(model_name)
        blocks_args = deepcopy(blocks)
        global_params = GlobalParams()
        for key, value in args_dict.items():
            setattr(global_params, key, value)

        return cls(in_channels=in_channels,
                   blocks_args=blocks_args,
                   global_params=global_params,
                   )
