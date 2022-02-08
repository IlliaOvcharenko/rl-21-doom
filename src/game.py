import torch
import torchvision
import cv2

import numpy as np
import vizdoom as vzd

from collections import deque
from itertools import product
from pathlib import Path
from tqdm.cli import tqdm

from src.data import RLDataset

def init_game(scenario, random_state):
    scenarios_folder = Path(vzd.scenarios_path)

    assert scenario in ["basic", "deadly_corridor"]
    if scenario == "basic":
        config_file_path = scenarios_folder / "simpler_basic.cfg"
    elif scenario == "deadly_corridor":
        config_file_path = scenarios_folder / "deadly_corridor.cfg"

    game = vzd.DoomGame()
    game.load_config(str(config_file_path))
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()

    n = game.get_available_buttons_size()
    actions = [list(a) for a in product([0, 1], repeat=n)]

    game.set_seed(random_state)
    return game, actions


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
        else:
            state = RLDataset.state_preproc(state)
            state = state.unsqueeze(0).to(device)

            q_values = model(state)
            _, action_idx = torch.max(q_values, dim=1)
            action_idx = int(action_idx.item())
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

        total_reward = 0
        if done:
            total_reward = self.game.get_total_reward()
            self.reset()

        return reward, total_reward, done


    def run_episodes(self, model, n_episodes):
        model = model.eval()
        model =  model.cpu()
        episodes_info = [{ "frames": [], "reward": 0.0 } for _ in range(n_episodes)]
        for episode_idx in tqdm(range(n_episodes), desc="play episodes"):
            self.reset()
            done = False
            while not done:
                reward, total_reward, done = self.play_step(model, 0.0, "cpu")
                episodes_info[episode_idx]["frames"] += list(self.cur_frames)
            episodes_info[episode_idx]["reward"] = total_reward

        return episodes_info



# if __name__ == "__main__":
#     import pytorch_lightning as pl
#     from src.data import ExperienceReplayBuffer
#     from src.models import DummyQNet

#     random_state = 42
#     pl.seed_everything(random_state)

#     TODO init game
#     game, actions = init_game("deadly_corridor")
#     game, actions = init_game("basic", random_state)

#     # TODO init replay buffer
#     exp_replay_buffer = ExperienceReplayBuffer(10000)

#     # TODO init Agent with ReplayBuffer and Doom game as a parasm
#     agent = Agent(game, actions, exp_replay_buffer)

#     model = DummyQNet(len(actions))
#     record(model, agent)


