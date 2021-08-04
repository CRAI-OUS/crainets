import torch

from crainets.models import UNet, ResXUNet

inp = torch.rand(size=(2, 2, 256, 256))

unet = UNet(
    n_channels=2,
    n_classes=3,
    n=32,
    norm='batch_norm',
    bilinear=True,
    )
pred = unet(inp)
print(pred.shape)

resxunet = ResXUNet(
    n_channels=2,
    n_classes=3,
    n=32,
    n_repeats=1,
    groups=8,
    bias=True,
    ratio=1./16,
    norm='instance_norm',
    activation=torch.nn.ReLU(inplace=True),
    )

pred = resxunet(inp)
print(pred.shape)
