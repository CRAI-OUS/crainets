from testing.data_loader import train_loader, test_loader
from testing import config
from babyjesus.trainer.trainer import Trainer
from babyjesus.models.efficientnet import EfficientNet
from babyjesus.essentials.multi_loss import MultiLoss
from babyjesus.essentials.multi_metric import MultiMetric

from collections.abc import Iterable
import torch

# #TODO include more tests for other types of efficient net


def test_efficientnet():
    assert isinstance(train_loader, Iterable), 'train_loader is not an iterable'
    assert isinstance(config.TRAIN_CONFIG, dict)

    # specifiy the nec config
    model = EfficientNet.from_name(in_channels=3, model_name='efficientnet-c16')
    loss = [(1, torch.nn.CrossEntropyLoss())]
    loss = MultiLoss(losses=loss)

    metrics = MultiMetric(config.METRICS)
    # get subsample from the dataloader

    trainer = Trainer(
        model=model,
        loss_function=loss,
        metric_ftns=metrics,
        config=config.TRAIN_CONFIG,
        data_loader=train_loader,
        valid_data_loader=test_loader,
        seed=666
    )

    trainer.train()
