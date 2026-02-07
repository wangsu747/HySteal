


import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pettingzoo.mpe import simple_tag_v3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hysteal.grf_wrapper import GRFParallelEnv
from hysteal.overcooked_wrapper import OvercookedParallelEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class NetCfg:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    n_layers: int
    use_agent_id: bool
    n_agents: int
    agent_emb_dim: int


class SimplePolicyNet(nn.Module):
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

    def forward(self, obs: torch.Tensor, agent_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.use_agent_id:
            if agent_id is None:
                raise ValueError("agent_id required when use_agent_id=True")
            x = torch.cat([obs, self.agent_emb(agent_id)], dim=-1)
        else:
            x = obs
        return self.net(x)


def load_model_auto(ckpt_path: str, device: torch.device, force_torchscript: bool):
    if force_torchscript:
        model = torch.jit.load(ckpt_path, map_location=device)
        model.eval()
        return model, None, None, []

    obj = None
    try:
        obj = torch.load(ckpt_path, map_location=device)
    except Exception:
        model = torch.jit.load(ckpt_path, map_location=device)
        model.eval()
        return model, None, None, []

    if isinstance(obj, nn.Module):
        obj.eval()
        return obj, None, None, []

    if isinstance(obj, dict):
        state_dict = obj.get("state_dict", None)
        if state_dict is None:
            state_dict = obj.get("policy_state_dict", None)
        if state_dict is None:
            raise ValueError("Checkpoint dict missing state_dict/policy_state_dict")

        cfg = NetCfg(
            obs_dim=int(obj["obs_dim"]),
            action_dim=int(obj["action_dim"]),
            hidden_dim=int(obj.get("hidden_dim", 256)),
            n_layers=int(obj.get("n_layers", 2)),
            use_agent_id=bool(obj.get("use_agent_id", False)),
            n_agents=int(obj.get("n_agents", len(obj.get("agent_order", [])) or 1)),
            agent_emb_dim=int(obj.get("agent_emb_dim", 16)),
        )
        pi = SimplePolicyNet(cfg).to(device)
        pi.load_state_dict(state_dict, strict=True)
        pi.eval()
        agent_order = obj.get("agent_order", [])
        return pi, cfg.obs_dim, cfg.use_agent_id, agent_order

    raise ValueError(f"Unsupported checkpoint object type: {type(obj)}")


def build_obs_vec(obs_map: Dict[str, np.ndarray], agent_name: str, obs_dim: Optional[int]) -> np.ndarray:
    vec = np.asarray(obs_map[agent_name], dtype=np.float32).reshape(-1)
    if obs_dim is None:
        return vec
    out = np.zeros((obs_dim,), dtype=np.float32)
    L = min(obs_dim, vec.shape[0])
    out[:L] = vec[:L]
    if agent_name == "agent_0" and obs_dim >= 2:
        out[obs_dim - 2] = out[0]
        out[obs_dim - 1] = out[1]
    return out


def pick_action(model, obs_vec: np.ndarray, agent_id: Optional[int], device: torch.device, use_agent_id: Optional[bool]):
    x = torch.from_numpy(obs_vec).unsqueeze(0).to(device)
    aid = None
    if use_agent_id:
        aid = torch.tensor([int(agent_id)], dtype=torch.long, device=device)
    with torch.no_grad():
        try:
            if use_agent_id:
                logits = model(x, aid)
            else:
                logits = model(x)
        except TypeError:
            logits = model(x)
        act = int(torch.argmax(logits, dim=-1).item())
    return act


def eval_mpe(model, obs_dim, use_agent_id, agent_order, args, device):
    if not agent_order:
        agent_order = [f"adversary_{i}" for i in range(args.num_adversaries)] + ["agent_0"]

    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}
    a2i = {a: i for i, a in enumerate(agent_order)}

    for ep in range(args.episodes):
        env = simple_tag_v3.parallel_env(
            max_cycles=args.max_cycles,
            continuous_actions=False,
            num_adversaries=args.num_adversaries,
            num_good=args.num_good,
        )
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = {a: 0.0 for a in agent_order}

        while True:
            if not obs:
                break
            act_dict = {}
            for a in agent_order:
                if a not in obs:
                    continue
                x = build_obs_vec(obs, a, obs_dim)
                act_dict[a] = pick_action(model, x, a2i[a], device, use_agent_id)

            if not act_dict:
                break

            obs, rewards, terms, truncs, _ = env.step(act_dict)
            for a, r in rewards.items():
                if a in ep_reward:
                    ep_reward[a] += float(r)
            if any(terms.values()) or any(truncs.values()):
                break

        env.close()
        per_ep_total.append(float(sum(ep_reward.values())))
        for a in agent_order:
            per_agent_total[a].append(float(ep_reward[a]))

    return {
        "ep_return_mean": float(np.mean(per_ep_total)),
        "ep_return_std": float(np.std(per_ep_total)),
        "per_agent_mean": {a: float(np.mean(v)) for a, v in per_agent_total.items()},
        "num_episodes": args.episodes,
    }


def eval_grf(model, obs_dim, use_agent_id, agent_order, args, device):
    env = GRFParallelEnv(
        env_name=args.grf_env_name,
        n_left=args.grf_n_left,
        n_right=args.grf_n_right,
        rewards=args.grf_rewards,
        render=False,
        episode_length=args.episode_length,
        shaping=args.grf_use_shaping,
    )

    if not agent_order:
        agent_order = list(env.agents)
    a2i = {a: i for i, a in enumerate(agent_order)}
    if obs_dim is None:
        obs_dim = int(env.obs_dim)

    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}

    for ep in range(args.episodes):
        obs = env.reset()
        ep_reward = {a: 0.0 for a in agent_order}

        while True:
            act_dict = {}
            for a in agent_order:
                x = np.asarray(obs[a], dtype=np.float32).reshape(-1)
                if obs_dim is not None:
                    vec = np.zeros((obs_dim,), dtype=np.float32)
                    L = min(obs_dim, x.shape[0])
                    vec[:L] = x[:L]
                else:
                    vec = x
                act_dict[a] = pick_action(model, vec, a2i[a], device, use_agent_id)

            obs, rewards, dones, _ = env.step(act_dict)
            for a, r in rewards.items():
                if a in ep_reward:
                    ep_reward[a] += float(r)
            if all(dones.values()):
                break

        per_ep_total.append(float(sum(ep_reward.values())))
        for a in agent_order:
            per_agent_total[a].append(float(ep_reward[a]))

    env.close()
    return {
        "ep_return_mean": float(np.mean(per_ep_total)),
        "ep_return_std": float(np.std(per_ep_total)),
        "per_agent_mean": {a: float(np.mean(v)) for a, v in per_agent_total.items()},
        "num_episodes": args.episodes,
    }


def eval_overcooked(model, obs_dim, use_agent_id, agent_order, args, device):
    env = OvercookedParallelEnv(
        layout_name=args.overcooked_layout,
        episode_length=args.episode_length,
        shape_reward=args.overcooked_shape_reward,
        render=False,
        quiet=True,
    )

    if not agent_order:
        agent_order = list(env.agents)
    a2i = {a: i for i, a in enumerate(agent_order)}
    if obs_dim is None:
        obs_dim = int(env.obs_dim)

    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}

    for ep in range(args.episodes):
        obs = env.reset()
        ep_reward = {a: 0.0 for a in agent_order}

        while True:
            act_dict = {}
            for a in agent_order:
                x = np.asarray(obs[a], dtype=np.float32).reshape(-1)
                if obs_dim is not None:
                    vec = np.zeros((obs_dim,), dtype=np.float32)
                    L = min(obs_dim, x.shape[0])
                    vec[:L] = x[:L]
                else:
                    vec = x
                act_dict[a] = pick_action(model, vec, a2i[a], device, use_agent_id)

            obs, rewards, dones, _ = env.step(act_dict)
            for a, r in rewards.items():
                if a in ep_reward:
                    ep_reward[a] += float(r)
            if all(dones.values()):
                break

        per_ep_total.append(float(sum(ep_reward.values())))
        for a in agent_order:
            per_agent_total[a].append(float(ep_reward[a]))

    env.close()
    return {
        "ep_return_mean": float(np.mean(per_ep_total)),
        "ep_return_std": float(np.std(per_ep_total)),
        "per_agent_mean": {a: float(np.mean(v)) for a, v in per_agent_total.items()},
        "num_episodes": args.episodes,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, required=True, choices=["simple_tag_v3", "grf", "overcooked"])
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--agent_order", type=str, default="")
    p.add_argument("--torchscript", action="store_true")
    p.add_argument("--save_json", type=str, default="")

    p.add_argument("--max_cycles", type=int, default=25)
    p.add_argument("--num_adversaries", type=int, default=3)
    p.add_argument("--num_good", type=int, default=1)

    p.add_argument("--grf_env_name", type=str, default="academy_empty_goal_close")
    p.add_argument("--grf_n_left", type=int, default=2)
    p.add_argument("--grf_n_right", type=int, default=0)
    p.add_argument("--grf_rewards", type=str, default="scoring,checkpoints")
    p.add_argument("--grf_use_shaping", action="store_true")

    p.add_argument("--overcooked_layout", type=str, default="cramped_room")
    p.add_argument("--overcooked_shape_reward", action="store_true")

    p.add_argument("--episode_length", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    model, obs_dim, use_agent_id, agent_order_ckpt = load_model_auto(args.ckpt_path, device, args.torchscript)

    agent_order = []
    if args.agent_order:
        agent_order = [x.strip() for x in args.agent_order.split(",") if x.strip()]
    elif agent_order_ckpt:
        agent_order = list(agent_order_ckpt)

    if args.env == "simple_tag_v3":
        result = eval_mpe(model, obs_dim, use_agent_id, agent_order, args, device)
    elif args.env == "grf":
        result = eval_grf(model, obs_dim, use_agent_id, agent_order, args, device)
    else:
        result = eval_overcooked(model, obs_dim, use_agent_id, agent_order, args, device)

    print(json.dumps(result, indent=2))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
