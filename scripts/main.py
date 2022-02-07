import sys,os
sys.path.append(os.getcwd())

import torch
import cv2

import numpy as np
import vizdoom as vzd
import pytorch_lightning as pl

from collections import deque
from itertools import product
from pathlib import Path
from fire import Fire

from src.models import (DuelQNet,
                        DummyQNet)
from src.frameio import (save_gif,
                         save_video)


def init_game(scenario, random_state):
    scenarios_folder = Path(vzd.scenarios_path)

    assert scenario in ["basic", "deadly_corridor"]
    if scenario == "basic":
        config_file_path = scenarios_folder / "simpler_basic.cfg"
    elif scenario == "deadly_corridor":
        config_file_path = scenarios_folder / "deadly_corridor.cfg"

    game = vzd.DoomGame()
    game.load_config(str(config_file_path))
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    n = game.get_available_buttons_size()
    actions = [list(a) for a in product([0, 1], repeat=n)]

    game.set_seed(random_state)
    return game, actions


class ExperienceReplayBuffer:
    """
    implement replay buffer logic
    experience struct: (state, action, reward, next_state, done)
    """

    def __init__(self, mem_len):
        self.mem = deque(maxlen=mem_len)

    def append(self, item):
        return self.mem.append(item)

    def sample(self, batch_size):
        # TODO non uniform sampling from replay buffer (prioritized)
        return np.random.choice(self.buffer, batch_size, replace=False)


class StatesDeque:
    def __init__(self, stack_size, screen_shape):
        self.stack_size = stack_size
        self.screen_shape = screen_shape
        self.mem = None
        self.reset()

    def reset(self):
        self.mem = deque([np.zeros(self.screen_shape) for i in range(self.stack_size)],
                         maxlen=self.stack_size)

    def append(self, state):
        self.mem.append(state)

    def stack(self):
        # print([f.shape for f in self.mem])
        return np.stack(self.mem)

    def append_and_stack(self, state):
        self.append(state)
        return self.stack()


class Agent:
    """ interact with doom env """
    def __init__(
            self,
            game,
            actions,
            exp_replay_buffer,
            frame_repeat=4,
            state_stack_size=4
    ):
        self.game = game

        self.exp_replay_buffer = exp_replay_buffer
        self.frame_repeat = frame_repeat

        self.actions = actions
        screen_shape = (self.game.get_screen_height(),
                        self.game.get_screen_width(),
                        self.game.get_screen_channels())
        # in case of grey scale remove channel dim
        if screen_shape[-1] == 1:
            screen_shape = screen_shape[:2]
        self.screen_shape = screen_shape

        self.states_deque = StatesDeque(state_stack_size, self.screen_shape)
        # TODO refactor cur_frames, it is strange
        self.cur_frames = deque(maxlen=frame_repeat)
        self.kills = None
        self.health = None
        self.ammo = None
        self.reset()

    def reset(self):
        self.game.new_episode()

        self.states_deque.reset()
        state = self.game.get_state().screen_buffer
        self.cur_frames.append(state)
        self.states_deque.append(state)

    def get_action(self, state, model, epsilon):
        if np.random.random() < epsilon:
            action_idx = np.random.choice(range(len(self.actions)))
            action = self.actions[action_idx]
        else:
            state = torch.tensor([state])

            # if device not in ["cpu"]:
                # state = state.cuda(device)

            q_values = model(state)
            _, action_idx = torch.max(q_values, dim=1)
            action_idx = int(action_idx.item())
            action = self.actions[action_idx]
        return action

    @torch.no_grad()
    def play_step(self, model, epsilon):
        state = self.states_deque.stack()
        action = self.get_action(state, model, epsilon)

        self.game.set_action(action)
        next_state = np.zeros(self.screen_shape)
        for _ in range(self.frame_repeat):
            self.game.advance_action()
            reward = self.game.get_last_reward()
            done = self.game.is_episode_finished()
            if done:
                break
            next_state = self.game.get_state().screen_buffer
            self.cur_frames.append(next_state)

        next_state = self.states_deque.append_and_stack(next_state)


        self.kills = self.game.get_game_variable(vzd.KILLCOUNT)
        self.health = self.game.get_game_variable(vzd.HEALTH)
        self.ammo = self.game.get_game_variable(vzd.AMMO2)

        self.exp_replay_buffer.append((state, action, reward, next_state, done))

        if done:
            self.reset()

        return reward, done


def record(model, agent):
    frames = list(agent.cur_frames)
    epsilon = 0.2
    done = False
    while not done:
        reward, done = agent.play_step(model, epsilon)
        frames += list(agent.cur_frames)

    save_gif("test.gif", frames, 10)
    save_video("test.mp4", frames, 10)


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
                 metrics,
                 replay_size,
                 frame_repeat,
                 state_stack_size,
                 random_state=42,):

        super().__init__()
        self.save_hyperparameters()
        pl.seed_everything(random_state)

        self.online_model = model_class(**model_kwargs)
        self.target_model = model_class(**model_kwargs)
        self.optimizer = optimizer_class(self.model.parameters(),
                                         **optimizer_kwargs)
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

        self.criterion = criterion
        self.monitor = monitor
        self.metrics = metrics


        self.game, self.actions = init_game("basic", random_state)
        self.exp_replay_buffer = ExperienceReplayBuffer(replay_size)
        self.agent = Agent(self.game, self.actions, self.exp_replay_buffer,
                          frame_repeat=frame_repeat, state_stack_size=state_stack_size)
        self.populate()


    def populate(self, n_steps):
        """ fill experience replay buffer before training """
        for _ in range(n_steps):
            self.agent.play_step(self.oline_model, epsilon=1.0)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        outs = self(imgs)
        loss = self.criterion(outs, targets)
        self.log("train_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        outs = self(imgs)
        loss = self.criterion(outs, targets)
        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 logger=True)

        scores = {}
        for metric_name, metric in  self.metrics.items():
            scores[f"val_{metric_name}"] = metric(outs, targets)

        self.log_dict(scores,
                      prog_bar=False,
                      on_step=False,
                      on_epoch=True,
                      logger=True)
        return loss

    def configure_optimizers(self):
        scheduler_info = {
            "scheduler": self.scheduler,
            "monitor": self.monitor,
            'interval': 'epoch',
        }
        return [self.optimizer], [scheduler_info]


def main():
    random_state = 42
    pl.seed_everything(random_state)

    # TODO init game
    # game, actions = init_game("deadly_corridor")
    game, actions = init_game("basic", random_state)

    # # TODO init replay buffer
    # exp_replay_buffer = ExperienceReplayBuffer(10000)

    # # TODO init Agent with ReplayBuffer and Doom game as a parasm
    # agent = Agent(game, actions, exp_replay_buffer)

    # model = DummyQNet(len(actions))
    # record(model, agent)

    # # TODO fill ReplayBuffer

    # model = DuelQNet(len(actions), 3, "usual")
    model = DuelQNet(len(actions), 3, "effnet")
    x = torch.rand(5, 3, 96, 128)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    Fire(main)

