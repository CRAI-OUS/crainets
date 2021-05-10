from testing.data_loader import train_loader, test_loader
from testing import config
from deepness.trainer.trainer import Trainer
from deepness.models.efficientnet import EfficientNet
from deepness.essentials.multi_loss import MultiLoss
from deepness.essentials.multi_metric import MultiMetric

from collections.abc import Iterable
import torch


def test_trainer():
    print(f'this is the type: {type(train_loader)}')
    assert isinstance(train_loader, Iterable), 'train_loader is not an iterable'
    assert isinstance(config.TRAIN_CONFIG, dict)
    print(f'the model is of the type: {type(EfficientNet)}')
    print(f'the metrics are: {config.METRICS}')

    model = EfficientNet.from_name(in_channels=3, model_name='efficientnet-b0')
    loss = [(1, torch.nn.L1Loss()), (1, torch.nn.MSELoss())]
    loss = MultiLoss(losses=loss)
    metrics = MultiMetric(config.METRICS)

    for idx, (x, y) in enumerate(train_loader):
        if idx == 0:
            breakpoint()
        else:
            break

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

