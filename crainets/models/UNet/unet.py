# External third party modules
import torch.nn as nn

# Internal modules
from .unet_blocks import DoubleConv, UpConv, DownConv, OutConv


class UNet(nn.Module):

    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 n: int = 128,
                 norm: str = 'batch_norm',
                 bilinear: bool = True):

        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, n, norm=norm)
        self.down1 = DownConv(n, 2*n, norm=norm)
        self.down2 = DownConv(2*n, 4*n, norm=norm)
        self.down3 = DownConv(4*n, 8*n, norm=norm)
        self.down4 = DownConv(8*n, 16*n // factor, norm=norm)

        self.up1 = UpConv(16*n, 8*n // factor, bilinear, norm=norm)
        self.up2 = UpConv(8*n, 4*n // factor, bilinear, norm=norm)
        self.up3 = UpConv(4*n, 2*n // factor, bilinear, norm=norm)
        self.up4 = UpConv(2*n, n, bilinear, norm=norm)
        self.outc = OutConv(n, n_classes)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
