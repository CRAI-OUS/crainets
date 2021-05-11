import torch
import torchvision
import testing.config as config
import torch.utils.data as data_utils

train = torchvision.datasets.CIFAR10(
                    config.TESTING, train=True, download=True,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()]))


test = torchvision.datasets.CIFAR10(
                    config.TESTING, train=False, download=True,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()]))

idx = torch.arange(1000)
trainx = data_utils.Subset(train, idx)
testx = data_utils.Subset(test, idx)

train_loader = torch.utils.data.DataLoader(
                        trainx,
                        batch_size=config.batch_size_train,
                        shuffle=True
                    )
test_loader = torch.utils.data.DataLoader(
                        testx,
                        batch_size=config.batch_size_test,
                        shuffle=True
                    )
