import numpy as np
import torch


class Buffer:


    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)
        self.triggered = np.zeros(capacity, dtype=bool)
        self.step_in_episode = np.zeros(capacity, dtype=np.int32)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done, triggered=False, step_in_episode=0):


        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done
        self.triggered[self._index] = bool(triggered)
        self.step_in_episode[self._index] = int(step_in_episode)

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]
        triggered = self.triggered[indices]
        step_in_episode = self.step_in_episode[indices]

        obs = torch.from_numpy(obs).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)
        triggered = torch.from_numpy(triggered).float().to(self.device)
        step_in_episode = torch.from_numpy(step_in_episode).long().to(self.device)

        return obs, action, reward, next_obs, done, triggered, step_in_episode

    def __len__(self):
        return self._size
