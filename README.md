# CRAI-Nets <img align="right" width="150" alt="20200907_104224" src="https://user-images.githubusercontent.com/29639563/125202990-9fcd9200-e276-11eb-8e00-bde211ebe0c1.png">

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Generic badge](https://img.shields.io/badge/contributions-welcome-<COLOR>.svg)](https://shields.io/)
[![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![License](https://img.shields.io/badge/license-BSD%204--Clause-red.svg)](https://shields.io/)

#### The CRAI-Nets Project
This is just another model-zoo and utility library combined for developing deep learning models. The main reasons for this project to exist is to avoid boilerplate code across projects, letting others tap in on your work, making benchmarking/expermenting easy and fast while also sticking to readibility and reproducibility. The goal of the project is to include as many useful models as possible and also smart customized metrics and loss functions. The project, as of now, is aimed towards computer vision, although contribution within NLP or RL is more than welcome.


#### Getting started

##### 0. Requirements
The library is platform agnostic although we strongly suggest to use Linux or Mac for ML development. We also suggest to use `poetry` or `pyenv` for dependency management unless you are on Win where Conda is the defacto(satans speed to you). Make sure to have python version 3.8 or later installed.


##### 1. Install the package
As recommended, use poetry to install the package by running:

```
$ poetry add crainets
```

##### 2. What you need to consider

The Trainer class you can use for simple benchmarking or fast expermenting expects mainly the following:

1. A model configuration dict containing hyperparameters 
2. A dict containing your loss functions
3. A dict containing your metrics (you can specify multiple)
4. Train and test data that you should prep in dataloader class that inherits from the pytorch `dataset` class
5. The model architecture imported from crainets model-zoo

We suggest to write your code modular such that configurations come from a `config.py` script and the dataloader comes from a `dataloader.py` script.

##### 3. Example

1. Lets write up two dataloaders that will lazy evaluate our data durng runtime when its batched for training. Cifar10 is used in this example and the only reason why is for brevity.

```python

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
                    config.DATA_PATH, train=True, download=True,
                    transform=transform)


test = torchvision.datasets.CIFAR10(
                    config.DATA_PATH, train=False, download=True,
                    transform=transform_test)

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
```


2. Now that we have our data, lets write up a config dict for our network to use.

```python
import os
import torch

ROOT = os.getcwd()
DATA_PATH = os.path.join('/data')
CHCKPT = os.path.join('/checkpoints')
batch_size_train = 100
batch_size_test = 50

TRAIN_CONFIG = {
        "n_gpu": 1,
        "optimizer": {
                "type": "Adam",
                "args": {
                    "lr": 1e-3,
                    "weight_decay": 0,
                    "amsgrad": True
                }
            },
            "loss": "nll_loss",
            "metrics": [
                "accuracy", "top_k_acc"
            ],
            "lr_scheduler": {
                "type": "StepLR",
                "args": {
                    "step_size": 500,
                    "gamma": 0.1
                }
            },
            "trainer": {
                "epochs": 2,
                "iterative": False,
                "iterations": 5,
                "images_pr_iteration": 100,
                "val_images_pr_iteration": 10,
                "save_dir": CHCKPT,
                "save_period": 5,
                "early_stop": 1
                }
            }

METRICS = {
        'CrossEntropy': torch.nn.CrossEntropyLoss()
            }
```

Note that we also included METRICS as a config in the script. We could define many more metrics in the dict than what is written in the example.

3. Now lets tie it all together in a controller script for running the network. We are going to use the sexy `efficient-net` in this example.

```python 
# Internal imports
from data_loader import train_loader, test_loader
from config import config

# CRAI-Nets imports
from crainets.trainer.trainer import Trainer
from crainets.models.efficientnet import EfficientNet
from crainets.essentials.multi_loss import MultiLoss
from crainets.essentials.multi_metric import MultiMetric

# specifiy the needed config
model = EfficientNet.from_name(in_channels=3, num_classes=10, model_name='efficientnet-b0')
loss = [(1, torch.nn.CrossEntropyLoss())]
loss = MultiLoss(losses=loss)
    
# Add metrics in the metrics dict from the config file
metrics = MultiMetric(config.METRICS)

# Instantiate zhe class
trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    config=config.TRAIN_CONFIG,
    data_loader=train_loader,
    valid_data_loader=test_loader,
    seed=666,
    accumulative_metrics=True
)

# Gut gut! Now run the network training und zmile!
trainer.train()
```

###### The project is mainly developed and maintained by CRAI at the university hospital of Oslo
