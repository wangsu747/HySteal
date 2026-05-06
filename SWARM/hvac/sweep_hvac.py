#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from HVAC.config import load_config
from HVAC.env import HVACCampusEnv
from HVAC.models import SharedActor
from noenv_simple.common import get_device


def parse_sizes(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("subset_sizes is empty.")
    return vals


def load_csv_subset(csv_path: str, max_episodes: int, horizon: int) -> Tuple[pd.DataFrame, List[int], List[str]]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "episode" not in df.columns or "t" not in df.columns or "agent" not in df.columns:
        raise ValueError("CSV must contain episode, t, agent.")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype(int)
    df["t"] = pd.to_numeric(df["t"], errors="coerce").astype(int)
    df = df[df["t"] < int(horizon)].copy()

    episodes = sorted(df["episode"].unique().tolist())
    picked = episodes[: min(max_episodes, len(episodes))]
    subset = df[df["episode"].isin(picked)].copy()
    agent_order = list(dict.fromkeys(subset["agent"].astype(str).tolist()))
    return subset, picked, agent_order


def summarize_reward_from_csv(df: pd.DataFrame, agent_order: List[str]) -> Dict:
    if "reward" not in df.columns:
        raise ValueError("CSV missing reward.")
    df = df.copy()
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)

    ep_total = df.groupby("episode")["reward"].sum()
    per_agent = df.groupby("agent")["reward"].mean()
    return {
        "ep_return_mean": float(ep_total.mean()) if len(ep_total) else 0.0,
        "ep_return_std": float(ep_total.std(ddof=0)) if len(ep_total) else 0.0,
        "per_step_agent_reward_mean": {a: float(per_agent.get(a, 0.0)) for a in agent_order},
        "num_episodes": int(df["episode"].nunique()),
        "num_rows": int(len(df)),
    }


@dataclass
class NetCfg:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    n_layers: int
    use_agent_id: bool
    n_agents: int
    agent_emb_dim: int


class SimplePolicy(nn.Module):
    def __init__(self, cfg: NetCfg):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.obs_dim
        if cfg.use_agent_id:
            self.agent_emb = nn.Embedding(cfg.n_agents, cfg.agent_emb_dim)
            in_dim += cfg.agent_emb_dim
        else:
            self.agent_emb = None
        layers = []
        cur = in_dim
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(cur, cfg.hidden_dim), nn.ReLU(inplace=True)]
            cur = cfg.hidden_dim
        layers.append(nn.Linear(cur, cfg.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, agent_id: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.use_agent_id:
            if agent_id is None:
                raise ValueError("agent_id is required when use_agent_id=True")
            x = torch.cat([obs, self.agent_emb(agent_id)], dim=-1)
        else:
            x = obs
        return self.net(x)


def load_simple_policy(ckpt_path: str, n_agents: int, device: torch.device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        state_dict = ckpt.get("policy_state_dict", None)
    if state_dict is None:
        raise ValueError(f"{ckpt_path} missing state_dict/policy_state_dict.")

    cfg = NetCfg(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        n_layers=int(ckpt["n_layers"]),
        use_agent_id=bool(ckpt["use_agent_id"]),
        n_agents=n_agents,
        agent_emb_dim=int(ckpt["agent_emb_dim"]),
    )
    pi = SimplePolicy(cfg).to(device)
    pi.load_state_dict(state_dict, strict=True)
    pi.eval()
    return {"policy": pi, "cfg": cfg, "agent_order": ckpt.get("agent_order", [])}


def load_shared_actor(ckpt_path: str, obs_dim: int, hidden_dim: int, num_actions: int, device: torch.device) -> SharedActor:
    ckpt = torch.load(ckpt_path, map_location=device)
    actor = SharedActor(obs_dim=obs_dim, hidden_dim=hidden_dim, num_actions=num_actions).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


def run_cmd(cmd: List[str], cwd: str) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def make_eval_env(config_path: str, horizon: int, seed: int) -> HVACCampusEnv:
    cfg = load_config(config_path)
    cfg.env.episode_steps = int(horizon)
    return HVACCampusEnv(cfg.env, seed=seed)


@torch.no_grad()
def run_random_policy_in_env(config_path: str, episodes: int, horizon: int, seed: int) -> Dict:
    env = make_eval_env(config_path, horizon=horizon, seed=seed)
    agent_order = list(env.agent_ids)
    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}
    per_ep_peak_power = []
    step_power = 0.0
    step_comfort = 0.0
    step_requested = 0.0
    step_curt = 0.0
    total_steps = 0

    for ep in range(episodes):
        obs = env.reset(random_start=True)
        done = False
        ep_reward = {a: 0.0 for a in agent_order}
        ep_peak_power = 0.0
        while not done:
            act_dict = {a: int(env.rng.randrange(env.num_actions)) for a in agent_order}
            obs, rewards, done, info = env.step(act_dict)
            for a, r in rewards.items():
                ep_reward[a] += float(r)
            ep_peak_power = max(ep_peak_power, float(info["total_power_kw"]))
            step_power += float(info["total_power_kw"])
            step_requested += float(info["total_requested_power_kw"])
            step_curt += float(info["curtailment_ratio"])
            step_comfort += sum(float(v["comfort_violation"]) for v in info["agents"].values()) / len(info["agents"])
            total_steps += 1
        per_ep_total.append(float(sum(ep_reward.values())))
        per_ep_peak_power.append(float(ep_peak_power))
        for a in agent_order:
            per_agent_total[a].append(float(ep_reward[a]))

    return {
        "ep_return_mean": float(np.mean(per_ep_total)),
        "ep_return_std": float(np.std(per_ep_total)),
        "per_agent_mean": {a: float(np.mean(v)) for a, v in per_agent_total.items()},
        "mean_peak_power_kw": float(np.mean(per_ep_peak_power)) if per_ep_peak_power else 0.0,
        "peak_power_std_kw": float(np.std(per_ep_peak_power)) if per_ep_peak_power else 0.0,
        "mean_power_kw": float(step_power / max(1, total_steps)),
        "mean_comfort_violation": float(step_comfort / max(1, total_steps)),
        "mean_requested_power_kw": float(step_requested / max(1, total_steps)),
        "mean_curtailment_ratio": float(step_curt / max(1, total_steps)),
        "num_episodes": int(episodes),
        "agent_order": agent_order,
    }


@torch.no_grad()
def run_shared_actor_in_env(
    actor: SharedActor,
    config_path: str,
    episodes: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> Dict:
    env = make_eval_env(config_path, horizon=horizon, seed=seed)
    agent_order = list(env.agent_ids)
    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}
    per_ep_peak_power = []
    step_power = 0.0
    step_comfort = 0.0
    step_requested = 0.0
    step_curt = 0.0
    total_steps = 0

    for ep in range(episodes):
        obs = env.reset(random_start=True)
        done = False
        ep_reward = {a: 0.0 for a in agent_order}
        ep_peak_power = 0.0
        while not done:
            stacked = torch.stack([obs[a] for a in agent_order], dim=0).to(device)
            dist = actor(stacked)
            actions = torch.argmax(dist.probs, dim=-1)
            act_dict = {a: int(actions[i].item()) for i, a in enumerate(agent_order)}
            obs, rewards, done, info = env.step(act_dict)
            for a, r in rewards.items():
                ep_reward[a] += float(r)
            ep_peak_power = max(ep_peak_power, float(info["total_power_kw"]))
            step_power += float(info["total_power_kw"])
            step_requested += float(info["total_requested_power_kw"])
            step_curt += float(info["curtailment_ratio"])
            step_comfort += sum(float(v["comfort_violation"]) for v in info["agents"].values()) / len(info["agents"])
            total_steps += 1
        per_ep_total.append(float(sum(ep_reward.values())))
        per_ep_peak_power.append(float(ep_peak_power))
        for a in agent_order:
            per_agent_total[a].append(float(ep_reward[a]))

    return {
        "ep_return_mean": float(np.mean(per_ep_total)),
        "ep_return_std": float(np.std(per_ep_total)),
        "per_agent_mean": {a: float(np.mean(v)) for a, v in per_agent_total.items()},
        "mean_peak_power_kw": float(np.mean(per_ep_peak_power)) if per_ep_peak_power else 0.0,
        "peak_power_std_kw": float(np.std(per_ep_peak_power)) if per_ep_peak_power else 0.0,
        "mean_power_kw": float(step_power / max(1, total_steps)),
        "mean_comfort_violation": float(step_comfort / max(1, total_steps)),
        "mean_requested_power_kw": float(step_requested / max(1, total_steps)),
        "mean_curtailment_ratio": float(step_curt / max(1, total_steps)),
        "num_episodes": int(episodes),
        "agent_order": agent_order,
    }


@torch.no_grad()
def run_simple_policy_in_hvac_env(
    policy: SimplePolicy,
    agent_order: List[str],
    config_path: str,
    episodes: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> Dict:
    env = make_eval_env(config_path, horizon=horizon, seed=seed)
    env_order = list(env.agent_ids)
    a2i = {a: i for i, a in enumerate(agent_order)}
    per_ep_total = []
    per_agent_total = {a: [] for a in env_order}
    per_ep_peak_power = []
    step_power = 0.0
    step_comfort = 0.0
    step_requested = 0.0
    step_curt = 0.0
    total_steps = 0

    for ep in range(episodes):
        obs = env.reset(random_start=True)
        done = False
        ep_reward = {a: 0.0 for a in env_order}
        ep_peak_power = 0.0
        while not done:
            act_dict = {}
            for a in env_order:
                x = obs[a].detach().cpu().numpy().astype(np.float32).reshape(-1)
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
                aid_t = torch.tensor([a2i[a]], dtype=torch.long, device=device) if policy.cfg.use_agent_id else None
                logits = policy(x_t, aid_t)
                act_id = int(torch.argmax(logits, dim=-1).item())
                act_dict[a] = act_id
            obs, rewards, done, info = env.step(act_dict)
            for a, r in rewards.items():
                ep_reward[a] += float(r)
            ep_peak_power = max(ep_peak_power, float(info["total_power_kw"]))
            step_power += float(info["total_power_kw"])
            step_requested += float(info["total_requested_power_kw"])
            step_curt += float(info["curtailment_ratio"])
            step_comfort += sum(float(v["comfort_violation"]) for v in info["agents"].values()) / len(info["agents"])
            total_steps += 1
        per_ep_total.append(float(sum(ep_reward.values())))
        per_ep_peak_power.append(float(ep_peak_power))
        for a in env_order:
            per_agent_total[a].append(float(ep_reward[a]))

    return {
        "ep_return_mean": float(np.mean(per_ep_total)),
        "ep_return_std": float(np.std(per_ep_total)),
        "per_agent_mean": {a: float(np.mean(v)) for a, v in per_agent_total.items()},
        "mean_peak_power_kw": float(np.mean(per_ep_peak_power)) if per_ep_peak_power else 0.0,
        "peak_power_std_kw": float(np.std(per_ep_peak_power)) if per_ep_peak_power else 0.0,
        "mean_power_kw": float(step_power / max(1, total_steps)),
        "mean_comfort_violation": float(step_comfort / max(1, total_steps)),
        "mean_requested_power_kw": float(step_requested / max(1, total_steps)),
        "mean_curtailment_ratio": float(step_curt / max(1, total_steps)),
        "num_episodes": int(episodes),
        "agent_order": env_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Train/evaluate BC/MAGAIL across expert data sizes on HVAC")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="HVAC/configs/campus_4agent.yml")
    parser.add_argument("--subset_sizes", type=str, default="200,400,600,800,1000")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--expert_policy_ckpt", type=str, default="")
    parser.add_argument("--train_bc", action="store_true")
    parser.add_argument("--train_magail", action="store_true")
    parser.add_argument("--bc_epochs", type=int, default=30)
    parser.add_argument("--magail_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--out_dir", type=str, default="HVAC/reward_sweep")
    args = parser.parse_args()

    sizes = parse_sizes(args.subset_sizes)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = os.path.join(args.out_dir, "data")
    ckpt_dir = os.path.join(args.out_dir, "ckpts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    random_result = run_random_policy_in_env(
        config_path=args.config,
        episodes=args.eval_episodes,
        horizon=args.horizon,
        seed=args.seed,
    )

    expert_result = None
    if args.expert_policy_ckpt.strip():
        cfg = load_config(args.config)
        tmp_env = HVACCampusEnv(cfg.env, seed=args.seed)
        actor = load_shared_actor(
            ckpt_path=args.expert_policy_ckpt,
            obs_dim=tmp_env.obs_dim,
            hidden_dim=cfg.train.hidden_dim,
            num_actions=tmp_env.num_actions,
            device=device,
        )
        expert_result = run_shared_actor_in_env(
            actor=actor,
            config_path=args.config,
            episodes=args.eval_episodes,
            horizon=args.horizon,
            seed=args.seed,
            device=device,
        )

    summary = {
        "setting": {
            "csv_path": args.csv_path,
            "config": args.config,
            "subset_sizes": sizes,
            "horizon": args.horizon,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
        },
        "random_env_reward": random_result,
        "expert_env_reward": expert_result,
        "results": [],
    }

    for k in sizes:
        subset_df, picked_eps, agent_order = load_csv_subset(args.csv_path, max_episodes=k, horizon=args.horizon)
        subset_csv = os.path.join(data_dir, f"expert_ep{k}_h{args.horizon}.csv")
        subset_df.to_csv(subset_csv, index=False)

        item = {
            "subset_episodes": int(len(picked_eps)),
            "requested_subset_episodes": int(k),
            "subset_csv": subset_csv,
            "agent_order": agent_order,
            "expert_csv_reward": summarize_reward_from_csv(subset_df, agent_order),
        }

        if args.train_bc:
            bc_ckpt = os.path.join(ckpt_dir, f"bc_ep{k}_h{args.horizon}.pt")
            run_cmd(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "noenv_simple", "train_bc.py"),
                    "--csv_path", subset_csv,
                    "--out_path", bc_ckpt,
                    "--use_agent_id",
                    "--agent_order", ",".join(agent_order),
                    "--epochs", str(args.bc_epochs),
                    "--batch_size", str(args.batch_size),
                    "--device", args.device,
                    "--seed", str(args.seed),
                ],
                cwd=PROJECT_ROOT,
            )
            obj = load_simple_policy(bc_ckpt, n_agents=len(agent_order), device=device)
            bc_eval = run_simple_policy_in_hvac_env(
                policy=obj["policy"],
                agent_order=agent_order if len(obj["agent_order"]) == 0 else obj["agent_order"],
                config_path=args.config,
                episodes=args.eval_episodes,
                horizon=args.horizon,
                seed=args.seed,
                device=device,
            )
            item["bc_env_reward"] = bc_eval
            item["bc_ckpt"] = bc_ckpt

        if args.train_magail:
            magail_ckpt = os.path.join(ckpt_dir, f"magail_ep{k}_h{args.horizon}.pt")
            run_cmd(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "noenv_simple", "train_magail.py"),
                    "--csv_path", subset_csv,
                    "--out_path", magail_ckpt,
                    "--use_agent_id",
                    "--agent_order", ",".join(agent_order),
                    "--epochs", str(args.magail_epochs),
                    "--batch_size", str(args.batch_size),
                    "--device", args.device,
                    "--seed", str(args.seed),
                ],
                cwd=PROJECT_ROOT,
            )
            obj = load_simple_policy(magail_ckpt, n_agents=len(agent_order), device=device)
            magail_eval = run_simple_policy_in_hvac_env(
                policy=obj["policy"],
                agent_order=agent_order if len(obj["agent_order"]) == 0 else obj["agent_order"],
                config_path=args.config,
                episodes=args.eval_episodes,
                horizon=args.horizon,
                seed=args.seed,
                device=device,
            )
            item["magail_env_reward"] = magail_eval
            item["magail_ckpt"] = magail_ckpt

        summary["results"].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    save_json = os.path.join(args.out_dir, "summary.json")
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] saved -> {save_json}")


if __name__ == "__main__":
    main()
