# External standard modules
import time
import sys
import json
from pathlib import Path
from typing import Union, Dict
from abc import abstractmethod
from datetime import datetime

# Third party modules
import py3nvml
import numpy as np
import torch

# Internal modules
from crainets.config.logger import get_logger
from crainets.essentials.multi_loss import MultiLoss
from crainets.essentials.multi_metric import MultiMetric
from crainets.essentials.multi_tracker import MetricTracker


class BaseTrainer:
    """
    Base class for all trainers

    Args:
       model: A torch.nn.Module class that contains a specific deep learning model
       loss_function: A dict or class type containing the loss functions
       metric_ftns: A dict or class type containing the model metrics
       optimizer: the optimizer function
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: MultiLoss,
                 metric_ftns: Union[MultiMetric, Dict[str, callable]],
                 config: Union[dict, str, Path],
                 seed: int = None,
                 device: str = None,
                 ):

        self.logger = get_logger(name=__name__)

        # Reproducibility is a good thing
        if isinstance(seed, int):
            torch.manual_seed(seed)

        # Read the config file from local disk if it is a path otherwise just set to a class object
        if not isinstance(config, (str, Path, dict)):
            raise TypeError(f'Input must be of type dict or str/pathlib.Path not {type(config)}')
        if isinstance(config, (str, Path)):
            with open(config, 'r') as inifile:
                self.logger.info('loading config file from local disk')
                self.config = json.load(inifile)
        else:
            self.config = config


        # setup GPU device if available, move model into configured device
        if device is None:
            self.device, device_ids = self.prepare_device(config['n_gpu'])
        else:
            self.device = torch.device(device)
            device_ids = list()

        # read in the model
        self.model = model.to(self.device)

        # TODO: Use DistributedDataParallel instead
        if len(device_ids) > 1 and config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_function = loss_function.to(self.device)

        if isinstance(metric_ftns, dict):  # dicts can't be sent to the gpu
            self.metrics_is_dict = True
            self.metric_ftns = metric_ftns
        else:  # MetricTracker class can be sent to the gpu
            self.metrics_is_dict = False
            self.metric_ftns = metric_ftns.to(self.device)

        trainer_cfg = config['trainer']
        self.epochs = trainer_cfg['epochs']
        self.save_period = trainer_cfg['save_period']

        self.iterative = bool(trainer_cfg['iterative'])
        self.iterations = int(trainer_cfg['iterations'])
        self.it = 'epoch' if self.iterative else 'iteration'

        self.start_epoch = 1

        self.checkpoint_dir = Path(trainer_cfg['save_dir']) / Path(datetime.today().strftime('%Y-%m-%d'))
        self.metric = MetricTracker(config=config)

        self.min_validation_loss = sys.float_info.max  # Minimum validation loss achieved, starting with the larges possible number

        # defining the optimizer
        optim = self.config['optimizer']
        optimizer = getattr(torch.optim, optim['type'])
        self.optimizer = optimizer(
                            model.parameters(),
                            lr=optim['args']['lr'],
                            weight_decay=optim['args']['weight_decay'],
                            amsgrad=optim['args']['amsgrad']
                            )

        # defining the learning rate scheduler
        lr_sched = self.config['lr_scheduler']
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_sched['type'])
        self.lr_scheduler = lr_scheduler(
                                    optimizer=self.optimizer,
                                    step_size=lr_sched['args']['step_size'],
                                    gamma=lr_sched['args']['gamma']
                                    )

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic after an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _train_iteration(self, epoch):
        """
        Training logic after an iteration, for large datasets
        :param epoch: Current iteration number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_iteration(self, epoch):
        """
        Validation logic after an iteration, for large datasets
        :param epoch: Current iteration number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        # Use iterations or epochs
        epochs = self.iterations if self.iterative else self.epochs

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start_time = time.time()
            loss_dict = self._train_epoch(epoch)
            val_dict = self._valid_epoch(epoch)
            epoch_end_time = time.time() - epoch_start_time


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            loss_dict['loss'] = np.mean(np.array(loss_dict['loss']))
            val_dict['loss'] = np.mean(np.array(val_dict['loss']))

            # save logged information regarding this epoch/iteration
            self.metric.training_update(loss=loss_dict, epoch=epoch)
            self.metric.validation_update(metrics=val_dict, epoch=epoch)

            self.logger.info(
                f"""
                Epoch/iteration {epoch} with validation completed in {epoch_end_time}
                run mean statistics:
                """
                )
            if hasattr(self.lr_scheduler, 'get_last_lr'):
                self.logger.info(f'Current learning rate: {self.lr_scheduler.get_last_lr()}')
            elif hasattr(self.lr_scheduler, 'get_lr'):
                self.logger.info(f'Current learning rate: {self.lr_sheduler.get_lr()}')

            # print logged informations to the screen
            # training loss
            loss = np.array(loss_dict['loss'])
            self.logger.info(f'Mean training loss: {loss}')

            if val_dict is not None:
                for key, valid in val_dict.items():
                    valid = np.array(valid)
                    self.logger.info(f'Mean validation {str(key)}: {np.mean(valid)}')

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)
            if val_dict['loss'] < self.min_validation_loss:
                self.min_validation_loss = val_dict['loss']
                self.save_checkpoint(epoch, best=True)

            self.logger.info('-----------------------------------')
        self.save_checkpoint(epoch, best=False)
        self.metric.write_to_file(path=self.checkpoint_dir / Path('model_statistics.json'))  # Save metrics at the end


    def prepare_device(self, n_gpu_use: int):
        """
        setup GPU device if available, move model into configured device
        Args:
            n_gpu_use (int): Number of GPU's to use
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                                """
                                There's no GPU available on this machine,
                                training will be performed on CPU.
                                """)
            n_gpu_use = n_gpu
        if n_gpu_use > n_gpu:
            self.logger.warning("""
                            The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine""")
            n_gpu_use = n_gpu

        free_gpus = py3nvml.get_free_gpus()

        list_ids = [i for i in range(n_gpu) if free_gpus[i]]
        n_gpu_use = min(n_gpu_use, len(list_ids))

        device = torch.device('cuda:{}'.format(list_ids[0]) if n_gpu_use > 0 else 'cpu')
        if device.type == 'cpu':
            self.logger.warning('current selected device is the cpu, you sure about this?')

        self.logger.info(f'Selected training device is: {device.type}:{device.index}')
        self.logger.info(f'The available gpu devices are: {list_ids}')

        return device, list_ids

    def save_checkpoint(self, epoch, best: bool = False):
        """
        Saving checkpoints at the given moment
        Args:
            epoch (int), the current epoch of the training
            bool (bool), save as best epoch so far, different naming convention
        """
        arch = type(self.model).__name__
        if self.lr_scheduler is not None:  # In case of None
            scheduler_state_dict = self.lr_scheduler.state_dict()
        else:
            scheduler_state_dict = None

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_state_dict,
            'config': self.config,
            'loss_func': str(self.loss_function),
            }

        if best:  # Save best case with different naming convention
            save_path = Path(self.checkpoint_dir) / Path('best_validation')
            filename = str(save_path / 'checkpoint-best.pth')
        else:
            save_path = Path(self.checkpoint_dir) / Path('epoch_' + str(epoch))
            filename = str(save_path / 'checkpoint-epoch{}.pth'.format(epoch))

        save_path.mkdir(parents=True, exist_ok=True)

        statics_save_path = save_path / Path('statistics.json')

        self.metric.write_to_file(path=statics_save_path)  # Save for every checkpoint in case of crash
        torch.save(state, filename)
        self.logger.info(f'Saving checkpoint: {filename} ...')

    def resume_checkpoint(self,
                          resume_model: Union[str, Path],
                          resume_metric: Union[str, Path]):
        """
        Resume from saved checkpoints
        Args:
            resume_model (str, pathlib.Path): Checkpoint path, either absolute or relative
        """
        if not isinstance(resume_model, (str, Path)):
            self.logger.warning('resume_model is not str or Path object but of type {type(resume_model)}, aborting previous checkpoint loading')
            return None

        if not Path(resume_model).is_file():
            self.logger.warning('resume_model object does not exist, ensure that {resume_model} is correct, aborting previous checkpoint loading')
            return None

        resume_model = str(resume_model)
        self.logger.info('Loading checkpoint: {resume_model} ...')

        try:
            checkpoint = torch.load(resume_model, map_location='cpu')
        except:
            checkpoint = torch.load(resume_model)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Different architecture from that given in the config, this may yield and exception while state_dict is loaded.')

        self.model.load_state_dict(checkpoint['state_dict'])

        self.model = self.model.to(self.device)

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning('Different optimizer from that given in the config, optimizer parameters are not resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load lr_scheduler state from checkpoint only when lr_scheduler type is not changed.
        if checkpoint['config']['lr_scheduler']['type'] != self.config['lr_scheduler']['type']:
            self.logger.warning('Different scheduler from that given in the config, scheduler parameters are not resumed.')
        elif self.lr_scheduler is None:
            self.logger.warning('lr_scheduler is None, scheduler parameters are not resumed.')
        else:
            if checkpoint['scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                self.logger.warning('lr_scheduler is saved as None, scheduler parameters cannot be resumed.')

        if resume_metric is None:
            self.logger.info('No path were given for prior statistics, cannot resume.')
            self.logger.info('New statistics will be written, and saved as regular.')
        else:
            self.metric.resume(resume_path=resume_metric)

        self.logger.info('Checkpoint loaded. Resume training from epoch {self.start_epoch}')

        self.checkpoint_dir = Path(resume_metric).parent.parent  # Ensuring the same main folder after resuming

        for key, value in self.metric[self.metric.VALIDATION_KEY].items():
            loss = np.mean(np.array(value['loss']))
            self.min_validation_loss = min(self.min_validation_loss, loss)
