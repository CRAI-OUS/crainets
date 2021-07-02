import torch
import torchvision
import testing.config as config

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

# idx = torch.arange(10000)
# trainx = data_utils.Subset(train, idx)
# testx = data_utils.Subset(test, idx)

train_loader = torch.utils.data.DataLoader(
                        train,
                        batch_size=config.batch_size_train,
                        shuffle=True
                    )
test_loader = torch.utils.data.DataLoader(
                        test,
                        batch_size=config.batch_size_test,
                        shuffle=True
                    )
