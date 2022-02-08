import torch
import torchvision
import cv2

import numpy as np
import vizdoom as vzd

from collections import deque
from pathlib import Path


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

