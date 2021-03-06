"""
Copyright (c) 2021, CRAI
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

# Standard modules
from typing import List, Tuple, Union

# Third party modules
import torch


class MultiLoss(torch.nn.Module):

    def __init__(self, losses: List[Tuple[Union[float, int], callable]]):
        super(MultiLoss, self).__init__()
        self.losses = losses

    def __repr__(self):
        return self.__class__.__name__ + '(losses={0})'.format(self.losses)

    def to(self, device):
        for i, (weight, loss_func) in enumerate(self.losses):
            self.losses[i] = (weight, loss_func.to(device=device))
        return self

    def cpu(self):
        for i, (weight, loss_func) in enumerate(self.losses):
            self.losses[i] = (weight, loss_func.cpu())
        return self

    def forward(self, x, y):
        loss = 0
        for (weight, loss_func) in self.losses:
            loss += weight*loss_func(x, y)

        return loss
