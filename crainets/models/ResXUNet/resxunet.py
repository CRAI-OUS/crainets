# External standard modules
from typing import Union

# External third party modules
import torch
import torch.nn.functional as F
import torch.nn as nn

# Internal modules
from ..blocks import (
    BasicBlock,
    Bottleneck,
    )

class ResXUNet(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n: int = 64,
                 n_repeats: int = 1,
                 groups: int = 32,
                 bias: bool = False,
                 ratio: float = 1./8,
                 norm: str = "batch_norm",
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.activation = nn.SiLU() if activation is None else activation


        self.input = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            self._norm(norm=norm, output=n, bias=bias),
            self.activation)

        self.inc = BasicBlock(
            channels=n,
            bias=bias,
            groups=groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )

        self.inc_bottle = nn.Sequential(*[Bottleneck(
            channels=n,
            mid_channels=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats // 2)])

        self.down1 = DownConvBlock(
            in_channels=n,
            out_channels=2*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation,
            )

        self.down1_basic = BasicBlock(
            channels=2*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )
        self.down1_bottle = nn.Sequential(*[Bottleneck(
            channels=2*n,
            mid_channels=2*n,
            groups=2*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats)])



        self.down2 = DownConvBlock(
            in_channels=2*n,
            out_channels=4*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation,
            )

        self.down2_basic = BasicBlock(
            channels=4*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )
        self.down2_bottle = nn.Sequential(*[Bottleneck(
            channels=4*n,
            mid_channels=4*n // 2,
            groups=2*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats)])




        self.down3 = DownConvBlock(
            in_channels=4*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation,
            )

        self.down3_basic = BasicBlock(
            channels=8*n,
            bias=bias,
            groups=4*groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )
        self.down3_bottle = nn.Sequential(*[Bottleneck(
            channels=8*n,
            mid_channels=8*n // 2,
            groups=4*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats)])



        self.down4 = DownConvBlock(
            in_channels=8*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation,
            )

        # Always guarantee two bottlenecks at the lowest level
        self.down4_bottle = nn.Sequential(*[Bottleneck(
            channels=8*n,
            mid_channels=8*n // 2,
            groups=4*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats + 2)])


        self.up4 = TransposeConvBlock(
            in_channels=8*n,
            out_channels=8*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation
            )




        self.up3_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=2*8*n,
                out_channels=8*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            self._norm(norm=norm, output=8*n, bias=bias),
            self.activation)


        self.up3_bottle = nn.Sequential(*[Bottleneck(
            channels=8*n,
            mid_channels=8*n // 2,
            groups=4*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats)])

        self.up3_basic = BasicBlock(
            channels=8*n,
            bias=bias,
            groups=4*groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )

        self.up3 = TransposeConvBlock(
            in_channels=8*n,
            out_channels=4*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation
            )




        self.up2_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=2*4*n,
                out_channels=4*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            self._norm(norm=norm, output=4*n, bias=bias),
            self.activation)

        self.up2_bottle = nn.Sequential(*[Bottleneck(
            channels=4*n,
            mid_channels=4*n // 2,
            groups=2*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats)])

        self.up2_basic = BasicBlock(
            channels=4*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )

        self.up2 = TransposeConvBlock(
            in_channels=4*n,
            out_channels=2*n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation
            )




        self.up1_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=2*2*n,
                out_channels=2*n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            self._norm(norm=norm, output=2*n, bias=bias),
            self.activation)

        self.up1_bottle = nn.Sequential(*[Bottleneck(
            channels=2*n,
            mid_channels=2*n,
            groups=2*groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats)])

        self.up1_basic = BasicBlock(
            channels=2*n,
            bias=bias,
            groups=2*groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )

        self.up1 = TransposeConvBlock(
            in_channels=2*n,
            out_channels=n,
            groups=1,
            bias=bias,
            norm=norm,
            activation=self.activation
            )



        self.out_channel = nn.Sequential(
            nn.Conv2d(
                in_channels=2*n,
                out_channels=n,
                kernel_size=3,
                stride=1,
                bias=False,
                padding=1),
            self._norm(norm=norm, output=n, bias=bias),
            self.activation)

        self.out_1 = BasicBlock(
            channels=n,
            bias=bias,
            groups=groups,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            )

        self.final_bottle = nn.Sequential(*[Bottleneck(
            channels=n,
            mid_channels=n,
            groups=groups,
            bias=bias,
            ratio=ratio,
            norm=norm,
            activation=self.activation,
            ) for i in range(n_repeats // 2 + 1)])

        self.outc = nn.Conv2d(in_channels=n, out_channels=n_classes, stride=1, kernel_size=1)

    def forward(self, x):

        x = self.input(x)
        x = self.inc(x)
        x1 = self.inc_bottle(x)


        x2 = self.down1(x1)
        x2 = self.down1_basic(x2)
        x2 = self.down1_bottle(x2)

        x3 = self.down2(x2)
        x3 = self.down2_basic(x3)
        x3 = self.down2_bottle(x3)

        x4 = self.down3(x3)
        x4 = self.down3_basic(x4)
        x4 = self.down3_bottle(x4)

        x = self.down4(x4)
        x = self.down4_bottle(x)
        x = self.up4(x)

        x = torch.cat([x, x4], dim=1)
        x = self.up3_channel(x)
        x = self.up3_bottle(x)
        x = self.up3_basic(x)
        x = self.up3(x)

        x = torch.cat([x, x3], dim=1)
        x = self.up2_channel(x)
        x = self.up2_bottle(x)
        x = self.up2_basic(x)
        x = self.up2(x)

        x = torch.cat([x, x2], dim=1)
        x = self.up1_channel(x)
        x = self.up1_bottle(x)
        x = self.up1_basic(x)
        x = self.up1(x)

        x = torch.cat([x, x1], dim=1)
        x = self.out_channel(x)
        x = self.out_1(x)
        x = self.final_bottle(x)

        return self.outc(x)


    def _norm(self, norm: str, output: int, bias: bool):
        if norm == "batch_norm":
            return nn.BatchNorm2d(
                num_features=output,
                affine=bias
                )
        elif norm == "instance_norm":
            return nn.InstanceNorm2d(
                num_features=output,
                affine=bias
                )
        else:
            return nn.LayerNorm(
                normalized_shape=1,
                elementwise_affine=bias,
                )




# Additional blocks to make making the network easier (avoid unnecessary repeated code)

class DownConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool,
                 norm: str,
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()
        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0,
            )

        self.norm = self._norm(norm=norm, output=out_channels, bias=bias)
        self.act = nn.SiLU() if activation is None else activation

    def forward(self, x: torch.Tensor):
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def _norm(self, norm: str, output: int, bias: bool):
        if norm == "batch_norm":
            return nn.BatchNorm2d(
                num_features=output,
                affine=bias
                )
        elif norm == "instance_norm":
            return nn.InstanceNorm2d(
                num_features=output,
                affine=bias
                )
        else:
            return nn.LayerNorm(
                normalized_shape=1,
                elementwise_affine=bias,
                )


class TransposeConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 groups: int,
                 bias: bool,
                 norm: str,
                 activation: Union[nn.Module, None] = None,
                 ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0,
            )

        self.norm = self._norm(norm=norm, output=out_channels, bias=bias)
        self.act = nn.SilU() if activation is None else activation

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def _norm(self, norm: str, output: int, bias: bool):
        if norm == "batch_norm":
            return nn.BatchNorm2d(
                num_features=output,
                affine=bias
                )
        elif norm == "instance_norm":
            return nn.InstanceNorm2d(
                num_features=output,
                affine=bias
                )
        else:
            return nn.LayerNorm(
                normalized_shape=1,
                elementwise_affine=bias,
                )

