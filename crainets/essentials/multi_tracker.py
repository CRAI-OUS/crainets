"""
Copyright (c) 2021, CRAI
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

# Standard modules
import json
from pathlib import Path
from typing import Union, Dict, Tuple, Optional

# External modules
import numpy as np
import matplotlib.pyplot as plt

# Internal modules
from crainets.config.logger import get_logger


class MetricTracker(object):
    """
    A simple class ment for storing and saving training loss history
    and validation metric history in json file
    """

    TRAINING_KEY = 'training'
    VALIDATION_KEY = 'validation'
    CONFIG_KEY = 'config'

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The config dict which initiates the network
        """

        self.logger = get_logger(name=__name__)

        self.results = dict()
        self.results[self.TRAINING_KEY] = dict()
        self.results[self.VALIDATION_KEY] = dict()
        self.results[self.CONFIG_KEY] = config
        self.iterative = bool(config['trainer']['iterative'])
        self.it = 'epoch' if self.iterative else 'iteration'

    def __getitem__(self, key):
        return self.results[key]

    def training(self):
        return self[self.TRAINING_KEY]

    def validation(self):
        return self[self.VALIDATION_KEY]

    def config(self):
        return self[self.CONFIG_KEY]

    def resume(self, resume_path: Union[str, Path]):
        """
        Resumes MetricTracker from previous state
        NB! Overwrites anything stored except the config dict
        Args:
            resume_path (str, pathlib.Path): The previous saved MetricTracker object
        """
        if not isinstance(resume_path, (str, Path)):
            TypeError('resume_path is not of type str or Path but {}'.format(type(resume_path)))

        if not Path(resume_path).is_file():
            self.logger.warning(f'{str(resume_path)} is not a file. Will not resume from MetricTracker instance.')
        with open(str(resume_path), 'r') as inifile:
            prev = json.load(inifile)

        if self.TRAINING_KEY not in prev.keys() or self.VALIDATION_KEY not in prev.keys():
            self.logger.warning('The given file does not have the training or validation key. Will not resume from prior checkpoint.')
            return

        if self.CONFIG_KEY in prev.keys():
            if prev[self.CONFIG_KEY] != self.results[self.CONFIG_KEY]:
                self.logger.warning('Non identical configs found, this instance will store the new config.')

        self.results[self.TRAINING_KEY].update(prev[self.TRAINING_KEY])
        self.results[self.VALIDATION_KEY].update(prev[self.VALIDATION_KEY])


    def training_update(self,
                        loss: Dict[str, list],
                        epoch: int):
        """
        Appends new training history
        Args:
            loss (list): np.ndarray, torch.Tensor): The loss history for this batch
            epoch (int): The epoch or iteration number, repeated numbers will overwrite
                         previous history
        """

        epoch = f'{self.it}_{epoch}'
        self.results[self.TRAINING_KEY][epoch] = loss

    def validation_update(self,
                          metrics: Dict[str, list],
                          epoch: int):
        """
        Args:
            metrics (dict): A dict matching the metric to the score for one/multiple metrics
            epoch (int): The epoch or iteration number, repeated numbers will overwrite
                         previous history
        """

        epoch = f'{self.it}_{epoch}'
        self.results[self.VALIDATION_KEY][epoch] = metrics

    def training_metric(self, epoch):
        return self.results[self.TRAINING_KEY][epoch]

    def validation_metric(self, epoch):
        return self.results[self.VALIDATION_KEY][epoch]

    def plot(self, show: bool = True, save_path: Optional[Union[str, Path]] = None):
        train = self.training()
        valid = self.validation()
        training = list()
        validation = list()
        x = np.arange(len(train.items()))

        for k, (i, j) in enumerate(train.items()):
            training.append(np.mean(np.array(j['loss'])))

        for k, (i, j) in enumerate(valid.items()):
            validation.append(np.mean(np.array(j['loss'])))

        cm = 1/2.54
        w = 17.6
        fig, ax = plt.subplots(figsize=(w*cm, 2*w/3*cm))
        plt.title("Validation and training loss")
        ax.plot(x, training, color='red', label='Train', linestyle='dashed', marker='x')
        ax.plot(x, validation, color='blue', label='Valid', linestyle='dashed', marker='x')

        ax.set_xlabel('Epoch', size=10)
        ax.set_ylabel('Loss', size=10)
        ax.legend(fontsize=10)
        plt.grid(which='major', alpha=0.25)
        if save_path is not None:
            plt.savefig(save_path,
                transparent=True,
                pad_inches=0,
                bbox_inches='tight',
                dpi=1200,
                )
        if show:
            plt.show()
        else:
            plt.close()

    def write_to_file(self, path: Union[str, Path]):
        """
        Writes MetricTracker to file
        Args:
            path (str, pathlib.Path): Path where the file is stored,
                                      remember to have .json suffix
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # Missing parents are quite the letdown
        path = str(path)

        with open(path, 'w') as outfile:
            json.dump(self.results, outfile, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        """
        Returns the metrics without the config
        """
        with open(str(path), 'r') as inifile:
            prev = json.load(inifile)
        metrics = cls(config=prev[cls.CONFIG_KEY])
        metrics.resume(path)
        return metrics

