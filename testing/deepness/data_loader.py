import torch
import torchvision
import config

train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST(
                                        config.ROOT, train=True, download=False,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()])),
                                batch_size=config.batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.MNIST(
                                            config.ROOT, train=False, download=False,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor()])),
                                batch_size=config.batch_size_test, shuffle=True)
