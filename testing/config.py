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
                "epochs": 20,
                "iterative": True,
                "iterations": 1,
                "images_pr_iteration": 1000,
                "val_images_pr_iteration": 500,
                "save_dir": CHCKPT,
                "save_period": 500,
                "early_stop": 10
                }
            }

METRICS = {
        'CrossEntropy': torch.nn.CrossEntropyLoss()
            }
