import torch
import torchvision
import testing.config as config
import torch.utils.data as data_utils

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train = torchvision.datasets.CIFAR10(
                    config.TESTING, train=True, download=True,
                    transform=transform)


test = torchvision.datasets.CIFAR10(
                    config.TESTING, train=False, download=True,
                    transform=transform_test)

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
