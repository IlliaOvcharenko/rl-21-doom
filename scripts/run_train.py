import sys,os
sys.path.append(os.getcwd())

import torch
import torchvision
import cv2

import numpy as np
import vizdoom as vzd
import pytorch_lightning as pl

from collections import deque
from itertools import product
from pathlib import Path
from fire import Fire
from tqdm.cli import tqdm

from src.train import DQNLightning

from src.models import (DuelQNet,
                        DummyQNet)
from src.utils import (compose_model_name,
                       get_next_model_version,
                       save_episodes)


def main():
    game_scenario="basic"
    # game_scenario="deadly_corridor"

    expr = DQNLightning(
        DuelQNet,
        {"in_channels": 3*4,  "encoder_mode": "usual"},

        torch.optim.Adam,
        {"lr": 0.00025},

        torch.optim.lr_scheduler.ReduceLROnPlateau,
        {"patience": 10000, "mode": "min", "factor": 0.6},

        monitor="train_loss",
        criterion=torch.nn.MSELoss(),

        replay_size=2000,
        frame_repeat=4,
        sync_rate=10,
        state_stack_size=4,
        discount=0.99,
        batch_size=256,
        steps_per_epoch=10000,

        epsilon_start=1.0,
        epsilon_decay=0.994,
        epsilon_min=0.1,

        game_scenario=game_scenario,
        random_state=42,
    )

    models_folder = Path("models")
    base_name = "dueldqn-basic"
    next_version = get_next_model_version(base_name, models_folder)
    model_full_name = compose_model_name(base_name, next_version)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=models_folder / model_full_name,
        filename="{epoch}-{step}-{train_loss:.4f}",
        mode="min",
        monitor="train_loss",
    )

    logs_folder = Path("logs")
    tb_logger = pl.loggers.TensorBoardLogger(
        logs_folder,
        name=base_name,
        version=next_version,
    )
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_logger],
        gpus=1,
        # max_epochs=20,
        max_epochs=1,
        deterministic=True,

        logger=tb_logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
    )

    trainer.fit(expr)
    episodes = expr.agent.run_episodes(expr.online_model, 3)
    figs_folder = Path("figs")
    save_episodes(episodes, model_full_name, figs_folder)


if __name__ == "__main__":
    Fire(main)

