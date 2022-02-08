import torch
import torchvision
import cv2

import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from tqdm.cli import tqdm

from src.models import (DuelQNet,
                        DummyQNet)
from src.data import (ExperienceReplayBuffer,
                      RLDataset)
from src.game import (init_game,
                      Agent)

class DQNLightning(pl.LightningModule):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optimizer_class,
                 optimizer_kwargs,
                 scheduler_class,
                 scheduler_kwargs,

                 criterion,
                 monitor,
                 # metrics,

                 replay_size=10000,
                 frame_repeat=4,
                 state_stack_size=4,
                 sync_rate=100,
                 discount=0.99,
                 batch_size=64,
                 steps_per_epoch=2000,

                 epsilon_start=1.0,
                 epsilon_decay=0.996,
                 epsilon_min=0.1,

                 game_scenario="basic",
                 random_state=42,):

        super().__init__()
        self.save_hyperparameters()
        pl.seed_everything(random_state)

        self.game, self.actions = init_game(game_scenario, random_state)
        self.exp_replay_buffer = ExperienceReplayBuffer(replay_size)
        self.agent = Agent(self.game, self.actions, self.exp_replay_buffer,
                          frame_repeat=frame_repeat, state_stack_size=state_stack_size)
        model_kwargs["action_space_size"] = len(self.actions)

        self.online_model = model_class(**model_kwargs)
        self.target_model = model_class(**model_kwargs)
        self.sync_rate = sync_rate

        self.optimizer = optimizer_class(self.online_model.parameters(),
                                         **optimizer_kwargs)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

        self.criterion = criterion
        self.monitor = monitor
        # self.metrics = metrics


        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.discount = discount
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        # self.total_reward = 0

        self.populate(batch_size * 2)


    def populate(self, n_steps):
        """ fill experience replay buffer before training """
        for _ in tqdm(range(n_steps), "fill replay buffer"):
            self.agent.play_step(self.online_model, epsilon=1.0, device="cpu")


    def forward(self, x):
        return self.online_model(x)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def calc_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        state_action_values = self.online_model(states) \
                                  .gather(1, actions.unsqueeze(-1)) \
                                  .squeeze(-1)

        with torch.no_grad():
            selected_actions = idx = self.online_model(next_states).argmax(1)
            next_state_values = self.target_model(next_states) \
                                    .gather(1, actions.unsqueeze(-1)) \
                                    .squeeze(-1)
            # next_state_values = self.target_model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.discount + rewards
        expected_state_action_values = expected_state_action_values.float()
        loss = self.criterion(state_action_values, expected_state_action_values)
        return loss

    def training_step(self, batch, batch_idx):
        device = self.get_device(batch)
        self.update_epsilon()

        reward, total_reward, done = self.agent.play_step(self.online_model, self.epsilon, device)

        if done:
            # print("total reward:", self.agent.game.get_total_reward())
            print("total reward:", total_reward)
        #     self.total_reward = self.agent.game.get_total_reward()

        loss = self.calc_loss(batch)

        logs = {
            "total_reward": torch.tensor(total_reward),
            "reward": torch.tensor(reward),
            "train_loss": loss,
            "epsilon": torch.tensor(self.epsilon)
        }
        self.log_dict(logs,
                      prog_bar=False,
                      on_step=True,
                      on_epoch=True,
                      logger=True)

        # Update target model
        # TODO or every training_epoch_end
        if self.global_step % self.sync_rate == 0:
            print("update target model")
            self.target_model.load_state_dict(self.online_model.state_dict())
        return loss

    def train_dataloader(self):
        dataset = RLDataset(self.exp_replay_buffer, self.steps_per_epoch)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
        )
        return dataloader

    def configure_optimizers(self):
        scheduler_info = {
            "scheduler": self.scheduler,
            "monitor": self.monitor,
            'interval': 'epoch',
        }
        return [self.optimizer], [scheduler_info]

    def get_device(self, batch):
        return batch[0].device.index if self.on_gpu else "cpu"

