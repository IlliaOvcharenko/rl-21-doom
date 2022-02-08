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


def main():
    # random_state = 42
    # pl.seed_everything(random_state)

    # TODO init game
    # game, actions = init_game("deadly_corridor")
    # game, actions = init_game("basic", random_state)

    # # TODO init replay buffer
    # exp_replay_buffer = ExperienceReplayBuffer(10000)

    # # TODO init Agent with ReplayBuffer and Doom game as a parasm
    # agent = Agent(game, actions, exp_replay_buffer)

    # model = DummyQNet(len(actions))
    # record(model, agent)

    # # TODO fill ReplayBuffer

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

        # game_scenario="deadly_corridor",
        game_scenario="basic",
        random_state=42,
    )

    models_folder = Path("models")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=models_folder,
        filename="{epoch}-{step}-{train_loss:.4f}",
        mode="min",
        monitor="train_loss",
    )

    logs_folder = Path("logs")
    tb_logger = pl.loggers.TensorBoardLogger(logs_folder)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_logger],
        gpus=1,
        max_epochs=20,
        deterministic=True,

        logger=tb_logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
    )

    trainer.fit(expr)


if __name__ == "__main__":
    Fire(main)

