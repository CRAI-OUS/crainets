"""
Copyright (c) 2021, CRAI
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

# Standard modules
from typing import Dict

# Third party modules
import torch


class MultiMetric(torch.nn.Module):

    def __init__(self, metrics: Dict[str, callable]):
        super(MultiMetric, self).__init__()
        self.metrics = metrics

    def __getitem__(self, key):
        return self.metrics[key]

    def __str__(self):
        return self.metrics

    def __repr__(self):
        return self.metrics

    def to(self, device):
        for key, metric in self.metrics.items():
            self.metrics[key] = metric.to(device=device)
        return self

    def cpu(self):
        for key, metric in self.metrics.items():
            self.metrics[key] = metric.cpu()
        return self

    def items(self):
        return self.metrics.items()
