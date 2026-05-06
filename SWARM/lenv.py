#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.dyn_csv import ResidualDynamicsMLP, DynCfg
from SWARM.utils.il_csv_adapter import normalize_il_dataframe


def _coerce_int_col(df: pd.DataFrame, col: str, default: int = 0) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)
    return df


def load_initial_joint_pool(
    csv_path: str,
    agent_order: List[str],
    obs_dim_per_agent: int,
    t0_only: bool = True,
) -> np.ndarray:
    df = pd.read_csv(csv_path, low_memory=False)
    df, _ = normalize_il_dataframe(df, obs_dim=obs_dim_per_agent, fill_value=0.0, ensure_next_obs=False)
    df = _coerce_int_col(df, "episode", default=0)
    df = _coerce_int_col(df, "t", default=0)
    df = df[df["agent"].isin(agent_order)].copy()

    if t0_only:
        df = df[df["t"] == 0].copy()

    agent_to_idx = {a: i for i, a in enumerate(agent_order)}
    joint_states = []
    for (_ep, _t), g in df.groupby(["episode", "t"], sort=True):
        if len(g) != len(agent_order):
            continue
        if g["agent"].nunique() != len(agent_order):
            continue
        g = g.sort_values(by="agent", key=lambda s: s.map(agent_to_idx))
        if list(g["agent"].values) != agent_order:
            continue
        obs = g[[f"obs_{i}" for i in range(obs_dim_per_agent)]].to_numpy(dtype=np.float32)
        joint_states.append(obs.reshape(-1))

    if len(joint_states) == 0:
        raise RuntimeError(f"No valid initial joint states found in {csv_path}")
    return np.stack(joint_states).astype(np.float32)


class LearnedDynamicsParallelEnv:
    """
    A light-weight PettingZoo environment
    """

    def __init__(
        self,
        dynamics_ckpt_path: str,
        init_csv_path: str,
        max_cycles: int,
        reset_from_t0_only: bool = True,
        reset_noise_sigma: float = 0.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        ckpt = torch.load(dynamics_ckpt_path, map_location=self.device)

        self.agent_order = list(ckpt["agent_order"])
        self.n_agents = int(ckpt["n_agents"])
        self.obs_dim_per_agent = int(ckpt["obs_dim_per_agent"])
        self.action_n = int(ckpt["action_n"])
        self.joint_state_dim = int(ckpt["joint_state_dim"])
        self.joint_action_dim = int(ckpt["joint_action_dim"])
        self.max_cycles = int(max_cycles)
        self.reset_noise_sigma = float(reset_noise_sigma)

        self.model = ResidualDynamicsMLP(
            DynCfg(
                in_dim=self.joint_state_dim + self.joint_action_dim,
                out_dim=self.joint_state_dim,
                hidden_dim=int(ckpt["hidden_dim"]),
                n_layers=int(ckpt["n_layers"]),
            )
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        self.initial_joint_pool = load_initial_joint_pool(
            csv_path=init_csv_path,
            agent_order=self.agent_order,
            obs_dim_per_agent=self.obs_dim_per_agent,
            t0_only=bool(reset_from_t0_only),
        )
        self._rng = np.random.default_rng(0)
        self._step_count = 0
        self._joint_state = None

    def _joint_to_obs_dict(self, joint_state: np.ndarray) -> Dict[str, np.ndarray]:
        mat = joint_state.reshape(self.n_agents, self.obs_dim_per_agent)
        return {a: mat[i].astype(np.float32).copy() for i, a in enumerate(self.agent_order)}

    def _act_dict_to_joint_onehot(self, act_dict: Dict[str, int]) -> np.ndarray:
        act_oh = np.zeros((self.n_agents, self.action_n), dtype=np.float32)
        for i, a in enumerate(self.agent_order):
            act_id = int(act_dict.get(a, 0))
            if 0 <= act_id < self.action_n:
                act_oh[i, act_id] = 1.0
        return act_oh.reshape(-1)

    def reset(self, seed: int = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        idx = int(self._rng.integers(low=0, high=len(self.initial_joint_pool)))
        s = self.initial_joint_pool[idx].copy()
        if self.reset_noise_sigma > 0:
            s = s + self._rng.normal(0.0, self.reset_noise_sigma, size=s.shape).astype(np.float32)
        self._joint_state = s.astype(np.float32)
        self._step_count = 0
        return self._joint_to_obs_dict(self._joint_state), {}

    def step(self, act_dict: Dict[str, int]):
        if self._joint_state is None:
            raise RuntimeError("Environment must be reset before step().")

        a = self._act_dict_to_joint_onehot(act_dict)
        st = torch.from_numpy(self._joint_state).unsqueeze(0).to(self.device)
        at = torch.from_numpy(a).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nxt, _ = self.model(st, at)
        self._joint_state = nxt.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self._step_count += 1

        obs = self._joint_to_obs_dict(self._joint_state)
        rewards = {a: 0.0 for a in self.agent_order}
        done = self._step_count >= self.max_cycles
        terminations = {a: False for a in self.agent_order}
        truncations = {a: bool(done) for a in self.agent_order}
        infos = {a: {} for a in self.agent_order}
        return obs, rewards, terminations, truncations, infos

    def close(self):
        return None


def make_learned_env_factory(
    dynamics_ckpt_path: str,
    init_csv_path: str,
    max_cycles: int,
    reset_from_t0_only: bool = True,
    reset_noise_sigma: float = 0.0,
    device: str = "cpu",
):
    def _make_env():
        return LearnedDynamicsParallelEnv(
            dynamics_ckpt_path=dynamics_ckpt_path,
            init_csv_path=init_csv_path,
            max_cycles=max_cycles,
            reset_from_t0_only=reset_from_t0_only,
            reset_noise_sigma=reset_noise_sigma,
            device=device,
        )
    return _make_env
