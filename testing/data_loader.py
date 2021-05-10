import torch
import torchvision
import testing.config as config

train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.CIFAR10(
                                        config.TESTING, train=True, download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()])),
                                batch_size=config.batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.CIFAR10(
                                            config.TESTING, train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor()])),
                                batch_size=config.batch_size_test, shuffle=True)
