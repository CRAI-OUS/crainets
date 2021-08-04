# Standard modules
from typing import Callable, Dict, Union
from collections import defaultdict

# Third party modules
import torch
import numpy as np

# Internal modules
from crainets.base.trainer_base import BaseTrainer
from crainets.essentials.multi_metric import MultiMetric
from crainets.essentials.multi_loss import MultiLoss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Union[MultiLoss, Callable],
                 metric_ftns: Union[MultiMetric, Dict[str, Callable]],
                 config: dict,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 log_step: int = None,
                 accumulative_metrics: bool = False,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         config=config,
                         seed=seed,
                         device=device)

        self.accumulative_metrics = accumulative_metrics

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.images_pr_iteration = int(config['trainer']['images_pr_iteration'])
        self.val_images_pr_iteration = int(config['trainer']['val_images_pr_iteration'])

        self.batch_size = data_loader.batch_size
        self.len_epoch = len(data_loader)*self.batch_size if not self.iterative else self.images_pr_iteration

        self.log_step = int(self.len_epoch/(self.batch_size*4)) if not isinstance(log_step, int) else log_step

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = defaultdict(list)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                self.logger.info(f'Train {self.it}: {epoch} {self._progress(batch_idx)} Loss: {loss}')

            if batch_idx*self.batch_size >= self.images_pr_iteration and self.iterative:
                break

        losses['loss_func'] = str(self.loss_function)

        return losses

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None

        self.model.eval()
        metrics = defaultdict(list)
        accuml = list()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                metrics['loss'].append(self.loss_function(output, target).item())

                if self.accumulative_metrics:
                    acc = (output.argmax(dim=1) == target).cpu().detach().type(torch.float32)
                    accuml.extend(list(acc))


                for key, metric in self.metric_ftns.items():
                    if self.metrics_is_dict:
                        metrics[key].append(metric(output.cpu(), target.cpu()).item())
                    else:
                        metrics[key].append(metric(output, target).item())

                if batch_idx*self.batch_size >= self.val_images_pr_iteration and self.iterative:
                    break

        if self.accumulative_metrics:
            metrics['accuracy'].append(float(np.mean(accuml)))

        return metrics

    def _train_iteration(self, iteration):
        """
        Training logic after an iteration, for large datasets
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        return self._train_epoch(epoch=iteration)

    def _valid_iteration(self, iteration):
        """
        Validation logic after an iteration, for large datasets
        :param epoch: Current iteration number
        """
        return self._valid_epoch(epoch=iteration)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx * self.data_loader.batch_size
            total = self.len_epoch
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
