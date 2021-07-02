from dataclasses import dataclass
from typing import Union, List, Tuple


@dataclass
class BlockArgs:
    num_repeat: int
    kernel_size: int
    stride: Union[int, Tuple[int, int]]
    expand_ratio: int
    input_filters: int
    output_filters: int
    se_ratio: Union[float, None] = 1./4
    id_skip: bool = True

# Efficientnet blocks
blocks = [
    BlockArgs(
        num_repeat=1,
        kernel_size=3,
        stride=[1, 1],
        expand_ratio=1,
        input_filters=32,
        output_filters=16,
    ),
    BlockArgs(
        num_repeat=2,
        kernel_size=3,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=16,
        output_filters=24,
    ),
    BlockArgs(
        num_repeat=2,
        kernel_size=5,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=24,
        output_filters=40,
    ),
    BlockArgs(
        num_repeat=3,
        kernel_size=3,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=40,
        output_filters=80,
    ),
    BlockArgs(
        num_repeat=3,
        kernel_size=5,
        stride=[1, 1],
        expand_ratio=6,
        input_filters=80,
        output_filters=112,
    ),
    BlockArgs(
        num_repeat=4,
        kernel_size=5,
        stride=[2, 2],
        expand_ratio=6,
        input_filters=112,
        output_filters=192,
    ),
    BlockArgs(
        num_repeat=1,
        kernel_size=3,
        stride=[1, 1],
        expand_ratio=6,
        input_filters=192,
        output_filters=320,
    ),
    ]

#NOTE! breach of DRY principle
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
    'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
    'efficientnet-b6', 'efficientnet-b7', 'efficientnet-b8',
    'efficientnet-l2',
)

@dataclass
class GlobalParams:
    width_coefficient: Union[float, None] = None
    depth_coefficient: Union[float, None] = None
    min_depth: Union[int, None] = None
    dropout_rate: float = 0.2
    batch_norm_momentum: float = 0.99
    batch_norm_epsilon: float = 0.001
    drop_connect_rate: float = 0.2
    depth_divisor: int = 8
    norm: str = 'batch_norm'  # Allowed. batch_norm, instance_norm, layer_norm
    num_classes: int = None


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.
    Args:
        model_name (str): Model name to be queried.
    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    #TODO: make this more dynamic for the enduser by letting it shift with a efficientnet-config file
    params_dict = {
        'efficientnet-b0':
            {'width_coefficient': 1.0, 'depth_coefficient': 1.0, 'resolution': 224, 'drop_connect_rate': 0.2},
        'efficientnet-b1':
            {'width_coefficient': 1.0, 'depth_coefficient': 1.1, 'resolution': 240, 'drop_connect_rate': 0.2},
        'efficientnet-b2':
            {'width_coefficient': 1.1, 'depth_coefficient': 1.2, 'resolution': 260, 'drop_connect_rate': 0.3},
        'efficientnet-b3':
            {'width_coefficient': 1.2, 'depth_coefficient': 1.4, 'resolution': 300, 'drop_connect_rate': 0.3},
        'efficientnet-b4':
            {'width_coefficient': 1.4, 'depth_coefficient': 1.8, 'resolution': 380, 'drop_connect_rate': 0.4},
        'efficientnet-b5':
            {'width_coefficient': 1.6, 'depth_coefficient': 2.2, 'resolution': 456, 'drop_connect_rate': 0.4},
        'efficientnet-b6':
            {'width_coefficient': 1.8, 'depth_coefficient': 2.6, 'resolution': 528, 'drop_connect_rate': 0.5},
        'efficientnet-b7':
            {'width_coefficient': 2.0, 'depth_coefficient': 3.1, 'resolution': 600, 'drop_connect_rate': 0.5},
        'efficientnet-b8':
            {'width_coefficient': 2.2, 'depth_coefficient': 3.6, 'resolution': 672, 'drop_connect_rate': 0.5},
        'efficientnet-l2':
            {'width_coefficient': 4.3, 'depth_coefficient': 5.3, 'resolution': 800, 'drop_connect_rate': 0.5},
        }

    return params_dict[model_name]
