#!/usr/bin/env python3

import argparse
import json
import random
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from pettingzoo.mpe import simple_spread_v3, simple_tag_v3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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
            x = torch.cat([obs, self.agent_emb(agent_id)], dim=-1)
        else:
            x = obs
        return self.net(x)


def build_obs_vec(obs_map: Dict[str, np.ndarray], agent_name: str, obs_dim: int, env_name: str) -> np.ndarray:
    vec = np.asarray(obs_map[agent_name], dtype=np.float32).reshape(-1)
    out = np.zeros((obs_dim,), dtype=np.float32)
    L = min(obs_dim, vec.shape[0])
    out[:L] = vec[:L]
    # Keep consistent with your training pipeline for agent_0.
    # Use the last two slots when obs_dim is larger than raw obs.
    if env_name == "simple_tag_v3" and agent_name == "agent_0" and obs_dim >= 2:
        out[obs_dim - 2] = out[0]
        out[obs_dim - 1] = out[1]
    return out


def make_parallel_env(
    env_name: str,
    max_cycles: int,
    num_adversaries: int,
    num_good: int,
    num_agents: int,
):
    if env_name == "simple_tag_v3":
        return simple_tag_v3.parallel_env(
            max_cycles=max_cycles,
            continuous_actions=False,
            num_adversaries=num_adversaries,
            num_good=num_good,
        )
    if env_name == "simple_spread_v3":
        return simple_spread_v3.parallel_env(
            N=num_agents,
            max_cycles=max_cycles,
            continuous_actions=False,
        )
    raise ValueError(f"Unsupported env_name: {env_name}")


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

    agent_order = ckpt.get("agent_order", [])
    return {"policy": pi, "cfg": cfg, "agent_order": agent_order}


def run_episodes_with_simple_policy(
    policy: SimplePolicy,
    agent_order: List[str],
    env_name: str,
    n_episodes: int,
    seed: int,
    max_cycles: int,
    num_adversaries: int,
    num_good: int,
    device: torch.device,
) -> Dict:
    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}
    a2i = {a: i for i, a in enumerate(agent_order)}

    for ep in range(n_episodes):
        env = make_parallel_env(
            env_name=env_name,
            max_cycles=max_cycles,
            num_adversaries=num_adversaries,
            num_good=num_good,
            num_agents=len(agent_order),
        )
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = {a: 0.0 for a in agent_order}

        while True:
            if not obs:
                break

            act_dict = {}
            for a in agent_order:
                if a not in obs:
                    continue
                x = build_obs_vec(obs, a, policy.cfg.obs_dim, env_name=env_name)
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
                if policy.cfg.use_agent_id:
                    aid_t = torch.tensor([a2i[a]], dtype=torch.long, device=device)
                else:
                    aid_t = None
                with torch.no_grad():
                    logits = policy(x_t, aid_t)
                    act_id = int(torch.argmax(logits, dim=-1).item())
                act_dict[a] = act_id

            if len(act_dict) == 0:
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
        "num_episodes": n_episodes,
    }


def run_episodes_with_magail_jointpolicy(
    joint_policy_path: str,
    agent_order: List[str],
    env_name: str,
    obs_dim: int,
    n_episodes: int,
    seed: int,
    max_cycles: int,
    num_adversaries: int,
    num_good: int,
    device: torch.device,
) -> Dict:
    jp = torch.load(joint_policy_path, map_location=device)
    jp = jp.to(device)
    jp.eval()
    if obs_dim <= 0:
        obs_dim = int(jp.config["agent"]["num_states"])

    per_ep_total = []
    per_agent_total = {a: [] for a in agent_order}
    n_agents = len(agent_order)

    for ep in range(n_episodes):
        env = make_parallel_env(
            env_name=env_name,
            max_cycles=max_cycles,
            num_adversaries=num_adversaries,
            num_good=num_good,
            num_agents=len(agent_order),
        )
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = {a: 0.0 for a in agent_order}

        while True:
            if not obs:
                break

            # pack states [n_agents, obs_dim]
            states = []
            for a in agent_order:
                if a not in obs:
                    states.append(np.zeros((obs_dim,), dtype=np.float32))
                else:
                    states.append(build_obs_vec(obs, a, obs_dim, env_name=env_name))
            s = torch.from_numpy(np.stack(states, axis=0)).to(device)  # [N,D]
            g = s.reshape(1, -1)  # [1,N*D]

            act_dict = {}
            with torch.no_grad():
                for i, a in enumerate(agent_order):
                    si = s[i].unsqueeze(0)  # [1,D]
                    act_vec, _ = jp.agent_policy[i].agent_get_action_log_prob(si, g)
                    # for discrete setting this is one-hot
                    act_id = int(torch.argmax(act_vec, dim=-1).item())
                    act_dict[a] = act_id

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
        "num_episodes": n_episodes,
    }


def main() -> None:
    p = argparse.ArgumentParser("Evaluate env rewards for simple_tag_v3 / simple_spread_v3")
    p.add_argument("--env_name", type=str, default="simple_tag_v3", choices=["simple_tag_v3", "simple_spread_v3"])
    p.add_argument("--agent_order", type=str, required=True, help="comma-separated, e.g. adversary_0,adversary_1,adversary_2,agent_0")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max_cycles", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num_adversaries", type=int, default=3)
    p.add_argument("--num_good", type=int, default=1)
    p.add_argument("--num_agents", type=int, default=3)

    p.add_argument("--magail_jointpolicy_path", type=str, default="")
    p.add_argument("--magail_obs_dim", type=int, default=0, help="<=0: auto from checkpoint")
    p.add_argument("--bc_ckpt_path", type=str, default="")
    p.add_argument("--offline_magail_like_ckpt_path", type=str, default="")
    p.add_argument("--save_json", type=str, default="")
    args = p.parse_args()

    agent_order = [x.strip() for x in args.agent_order.split(",") if x.strip()]
    if len(agent_order) == 0:
        raise ValueError("agent_order is empty.")
    device = get_device(args.device)
    set_seed(args.seed)

    summary = {
        "setting": {
            "env_name": args.env_name,
            "episodes": args.episodes,
            "max_cycles": args.max_cycles,
            "seed": args.seed,
            "agent_order": agent_order,
        },
        "results": {}
    }

    if args.magail_jointpolicy_path:
        summary["results"]["magail"] = run_episodes_with_magail_jointpolicy(
            joint_policy_path=args.magail_jointpolicy_path,
            agent_order=agent_order,
            env_name=args.env_name,
            obs_dim=args.magail_obs_dim,
            n_episodes=args.episodes,
            seed=args.seed,
            max_cycles=args.max_cycles,
            num_adversaries=args.num_adversaries,
            num_good=args.num_good,
            device=device,
        )
        print("[eval] MAGAIL done.")

    if args.bc_ckpt_path:
        obj = load_simple_policy(args.bc_ckpt_path, n_agents=len(agent_order), device=device)
        summary["results"]["bc"] = run_episodes_with_simple_policy(
            policy=obj["policy"],
            agent_order=agent_order if len(obj["agent_order"]) == 0 else obj["agent_order"],
            env_name=args.env_name,
            n_episodes=args.episodes,
            seed=args.seed,
            max_cycles=args.max_cycles,
            num_adversaries=args.num_adversaries,
            num_good=args.num_good,
            device=device,
        )
        print("[eval] BC done.")

    if args.offline_magail_like_ckpt_path:
        obj = load_simple_policy(args.offline_magail_like_ckpt_path, n_agents=len(agent_order), device=device)
        summary["results"]["offline_magail_like"] = run_episodes_with_simple_policy(
            policy=obj["policy"],
            agent_order=agent_order if len(obj["agent_order"]) == 0 else obj["agent_order"],
            env_name=args.env_name,
            n_episodes=args.episodes,
            seed=args.seed,
            max_cycles=args.max_cycles,
            num_adversaries=args.num_adversaries,
            num_good=args.num_good,
            device=device,
        )
        print("[eval] Offline MAGAIL-like done.")

    print(json.dumps(summary, indent=2))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[eval] saved -> {args.save_json}")


if __name__ == "__main__":
    main()
