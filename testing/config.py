import os
import torch

ROOT = os.getcwd()
TESTING = os.path.join('testing/data')
CHCKPT = os.path.join('testing/checkpoints')
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
