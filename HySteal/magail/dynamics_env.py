
import os
import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
@dataclass
class DynCfg:
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int
class ResidualDynamicsMLP(nn.Module):
    def __init__(self, cfg: DynCfg):
        super().__init__()
        layers = []
        cur = cfg.in_dim
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(cur, cfg.hidden_dim), nn.ReLU(inplace=True)]
            cur = cfg.hidden_dim
        layers += [nn.Linear(cur, cfg.out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        delta = self.net(x)
        return s + delta, delta
def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def _to_onehot(action_ids: np.ndarray, action_n: int) -> np.ndarray:
    action_ids = action_ids.astype(np.int64).reshape(-1)
    out = np.zeros((len(action_ids), action_n), dtype=np.float32)
    valid = (action_ids >= 0) & (action_ids < action_n)
    if np.any(valid):
        out[np.arange(len(action_ids))[valid], action_ids[valid]] = 1.0
    return out


def _split_joint_state(joint: np.ndarray, n_agents: int, obs_dim: int) -> Dict[str, np.ndarray]:
    out = {}
    for i in range(n_agents):
        s = joint[i * obs_dim:(i + 1) * obs_dim]
        out[f"agent_{i}"] = s.astype(np.float32).copy()
    return out


def _build_joint_action(action_dict: Dict[str, int], agent_order: List[str], action_n: int) -> np.ndarray:
    ids = []
    for a in agent_order:
        ids.append(int(action_dict.get(a, 0)))
    oh = _to_onehot(np.asarray(ids, dtype=np.int64), action_n)
    return oh.reshape(-1).astype(np.float32)


def load_dynamics_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    required = [
        "model_state_dict",
        "n_agents",
        "obs_dim_per_agent",
        "action_n",
        "joint_state_dim",
        "joint_action_dim",
        "hidden_dim",
        "n_layers",
    ]
    for k in required:
        if k not in ckpt:
            raise ValueError(f"Dynamics ckpt missing key: {k}")

    model = ResidualDynamicsMLP(
        DynCfg(
            in_dim=int(ckpt["joint_state_dim"] + ckpt["joint_action_dim"]),
            out_dim=int(ckpt["joint_state_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            n_layers=int(ckpt["n_layers"]),
        )
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, ckpt


class DynamicsEnv:
    def __init__(self, ckpt_path: str, episode_length: int, device: str = "auto", seed: int = 0):
        self.device = get_device(device)
        self.model, self.ckpt = load_dynamics_ckpt(ckpt_path, self.device)

        self.n_agents = int(self.ckpt["n_agents"])
        self.obs_dim = int(self.ckpt["obs_dim_per_agent"])
        self.action_n = int(self.ckpt["action_n"])
        self.agent_order = list(self.ckpt.get("agent_order", []))
        if not self.agent_order:
            self.agent_order = [f"agent_{i}" for i in range(self.n_agents)]

        self.episode_length = int(episode_length)
        self.step_count = 0

        self._rng = random.Random(int(seed))
        self._init_states = self.ckpt.get("init_joint_states", [])
        if isinstance(self._init_states, np.ndarray):
            self._init_states = self._init_states.tolist()

        if not self._init_states:
            self._init_states = [[0.0 for _ in range(int(self.ckpt["joint_state_dim"]))]]

        self._cur_state = None

    def reset(self):
        self.step_count = 0
        init = self._rng.choice(self._init_states)
        self._cur_state = np.asarray(init, dtype=np.float32).reshape(-1)
        return _split_joint_state(self._cur_state, self.n_agents, self.obs_dim), {}

    def step(self, action_dict: Dict[str, int]):
        joint_action = _build_joint_action(action_dict, self.agent_order, self.action_n)
        s = torch.from_numpy(self._cur_state).unsqueeze(0).to(self.device)
        a = torch.from_numpy(joint_action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_s, _ = self.model(s, a)
        self._cur_state = next_s.squeeze(0).detach().cpu().numpy().astype(np.float32)

        self.step_count += 1
        done = self.step_count >= self.episode_length

        obs = _split_joint_state(self._cur_state, self.n_agents, self.obs_dim)
        rewards = {a: 0.0 for a in self.agent_order}
        dones = {a: bool(done) for a in self.agent_order}
        infos = {a: {} for a in self.agent_order}
        return obs, rewards, dones, infos

    def close(self):
        pass
