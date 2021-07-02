from testing.data_loader import train_loader, test_loader
from testing import config
from crainet.trainer.trainer import Trainer
from crainet.models.efficientnet import EfficientNet
from crainet.essentials.multi_loss import MultiLoss
from crainet.essentials.multi_metric import MultiMetric

from collections.abc import Iterable
import torch

# #TODO include more tests for other types of efficient net


def test_efficientnet():
    assert isinstance(train_loader, Iterable), 'train_loader is not an iterable'
    assert isinstance(config.TRAIN_CONFIG, dict)

    # specifiy the nec config
    model = EfficientNet.from_name(in_channels=3, num_classes=10, model_name='efficientnet-b0')
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
        seed=666,
        accumulative_metrics=True
    )

    trainer.train()
