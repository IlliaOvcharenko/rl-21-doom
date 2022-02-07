import torch
import cv2

import numpy as np
import vizdoom as vzd
import pytorch_lightning as pl

from collections import deque
from itertools import product
from pathlib import Path
from fire import Fire


def init_game(scenario):
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
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    n = game.get_available_buttons_size()
    actions = [list(a) for a in product([0, 1], repeat=n)]

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

        # in case of grey scale remove channel dim
        if screen_shape[-1] == 1:
            screen_shape = screen_shape[:2]
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
    """
    interact with doom env
    """
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
        self.screen_shape = (self.game.get_screen_height(),
                             self.game.get_screen_width(),
                             self.game.get_screen_channels())

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

class DummyQNet:
    def __init__(self, action_space_size):
        # TODO add stratery like rand, fixed_ation, etc.
        # self.strategy = strategy
        self.action_space_size = action_space_size

    def __call__(self, x):
        batch_size = x.shape[0]
        return torch.rand((batch_size, self.action_space_size))


def save_gif(save_fn, frames, fps):
    from PIL import Image

    frames = [Image.fromarray(f) for f in frames]
    frame, *frames = frames
    ms_per_frame = int(1.0 / fps) * 1000
    frame.save(fp=save_fn, format='GIF', append_images=frames,
               save_all=True, duration=ms_per_frame, loop=0)


def record(model, agent):
    frames = list(reversed(agent.cur_frames))
    epsilon = 0.2
    done = False
    while not done:
        reward, done = agent.play_step(model, epsilon)
        frames += list(reversed(agent.cur_frames))

    save_gif("test.gif", frames, 10)

def main():
    pl.seed_everything(42)

    # TODO init game
    game, actions = init_game("deadly_corridor")

    # TODO init replay buffer
    exp_replay_buffer = ExperienceReplayBuffer(10000)

    # TODO init Agent with ReplayBuffer and Doom game as a parasm
    agent = Agent(game, actions, exp_replay_buffer)

    model = DummyQNet(len(actions))
    record(model, agent)

    # TODO fill ReplayBuffer


if __name__ == "__main__":
    Fire(main)

