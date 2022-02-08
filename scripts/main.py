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

    # def sample(self, batch_size):
    #     # TODO non uniform sampling from replay buffer (prioritized)
    #     # return np.random.choice(self.buffer, batch_size, replace=False)
    #     indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    #     states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))

    #     return (
    #         np.array(states),
    #         np.array(actions),
    #         np.array(rewards, dtype=np.float32),
    #         np.array(dones, dtype=np.bool),
    #         np.array(next_states),
    #     )

    def sample(self):
        # TODO non uniform sampling from replay buffer (prioritized)
        # return np.random.choice(self.buffer, batch_size, replace=False)
        idx = np.random.choice(range(len(self.mem)))
        item = self.mem[idx]
        # print(len(item))
        return item


# class RLDataset(torch.utils.data.datasets.IterableDataset):
#     def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
#         self.buffer = buffer
#         self.sample_size = sample_size

#     def __iter__(self) -> Tuple:
#         states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
#         for i in range(len(dones)):
#             yield states[i], actions[i], rewards[i], new_states[i], dones[i]

class RLDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, buffer, steps_per_epoch):
        self.buffer = buffer
        self.steps_per_epoch = steps_per_epoch

    def state_preproc(state):
        state = cv2.resize(state, (128, 96))
        state = np.transpose(state, (2, 0, 1))
        state = state.astype(float) / 255.0
        # state = torch.tensor(state).float()
        # state = torchvision.transforms.functional.to_tensor(state)
        # state = state / 255.0
        # state = state.double()
        state = torch.from_numpy(state).float()
        return state

    def __getitem__(self, idx):
        state, action, reward, next_state, dones = self.buffer.sample()
        state = RLDataset.state_preproc(state)
        next_state = RLDataset.state_preproc(next_state)
        return (state, action, reward, next_state, dones)

    def __len__(self):
        return self.steps_per_epoch


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
        # return np.stack(self.mem)
        return np.concatenate(self.mem, 2)

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

    def get_action(self, state, model, epsilon, device):
        if np.random.random() < epsilon:
            action_idx = np.random.choice(range(len(self.actions)))
            # action = self.actions[action_idx]
        else:
            state = RLDataset.state_preproc(state)
            state = state.unsqueeze(0).to(device)


            # if device not in ["cpu"]:
                # state = state.cuda(device)

            q_values = model(state)
            _, action_idx = torch.max(q_values, dim=1)
            action_idx = int(action_idx.item())
            # action = self.actions[action_idx]
        return action_idx

    @torch.no_grad()
    def play_step(self, model, epsilon, device):
        state = self.states_deque.stack()
        action = self.get_action(state, model, epsilon, device)

        self.game.set_action(self.actions[action])
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

                 random_state=42,):

        super().__init__()
        self.save_hyperparameters()
        pl.seed_everything(random_state)

        self.game, self.actions = init_game("basic", random_state)
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

        self.total_reward = 0

        self.populate(batch_size)


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
            # selected_actions = idx = self.online_model(next_states).argmax(1)
            # next_state_values = self.target_model(next_states) \
                                    # .gather(1, selected_actions)
            next_state_values = self.target_model(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.discount + rewards
        expected_state_action_values = expected_state_action_values.float()
        loss = self.criterion(state_action_values, expected_state_action_values)
        return loss

    def training_step(self, batch, batch_idx):
        device = self.get_device(batch)
        self.update_epsilon()

        reward, done = self.agent.play_step(self.online_model, self.epsilon, device)

        if done:
            self.total_reward = self.agent.game.get_total_reward()

        loss = self.calc_loss(batch)

        logs = {
            "total_reward": torch.tensor(self.total_reward),
            "reward": torch.tensor(reward),
            "train_loss": loss,
        }
        self.log_dict(logs,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True,
                      logger=True)

        # Update target model
        # TODO or every training_epoch_end
        if self.global_step % self.sync_rate == 0:
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
        {"lr": 0.01},

        torch.optim.lr_scheduler.ReduceLROnPlateau,
        {"patience": 10, "mode": "min", "factor": 0.6},

        monitor="train_loss",
        replay_size=10000,
        frame_repeat=4,
        sync_rate=2000,
        state_stack_size=4,
        discount=0.99,
        batch_size=64,
        steps_per_epoch=2000,

        epsilon_start=1.0,
        epsilon_decay=0.996,
        epsilon_min=0.1,

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
        max_epochs=10,
        deterministic=True,

        logger=tb_logger,
        log_every_n_steps=1,
        flush_logs_every_n_steps=1,
    )

    trainer.fit(expr)


if __name__ == "__main__":
    Fire(main)

