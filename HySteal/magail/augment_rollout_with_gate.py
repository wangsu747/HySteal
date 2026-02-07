
import argparse
import os
import random
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pettingzoo.mpe import simple_tag_v3
from magail.dynamics_env import DynamicsEnv


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class BCPolicy(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 use_agent_id: bool,
                 n_agents: int,
                 agent_emb_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_agent_id = use_agent_id
        self.n_agents = n_agents
        self.agent_emb_dim = agent_emb_dim

        in_dim = obs_dim
        if use_agent_id:
            self.agent_emb = nn.Embedding(n_agents, agent_emb_dim)
            in_dim += agent_emb_dim
        else:
            self.agent_emb = None

        layers = []
        cur = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            cur = hidden_dim
        layers.append(nn.Linear(cur, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, agent_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_agent_id:
            if agent_id is None:
                raise ValueError("agent_id required when use_agent_id=True")
            emb = self.agent_emb(agent_id)
            x = torch.cat([obs, emb], dim=-1)
        else:
            x = obs
        return self.net(x)


def load_bc(ckpt_path: str, device: torch.device) -> Tuple[BCPolicy, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt and "state_dict" in ckpt:
        ckpt["model_state_dict"] = ckpt["state_dict"]
    required = ["model_state_dict", "obs_dim", "action_dim", "use_agent_id",
                "hidden_dim", "n_layers", "agent_emb_dim", "agent_order"]
    for k in required:
        if k not in ckpt:
            raise ValueError(f"Checkpoint missing key: {k} ({ckpt_path})")

    agent_order = list(ckpt["agent_order"])
    n_agents = len(agent_order)

    model = BCPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        n_layers=int(ckpt["n_layers"]),
        use_agent_id=bool(ckpt["use_agent_id"]),
        n_agents=n_agents,
        agent_emb_dim=int(ckpt["agent_emb_dim"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "obs_dim": int(ckpt["obs_dim"]),
        "action_dim": int(ckpt["action_dim"]),
        "use_agent_id": bool(ckpt["use_agent_id"]),
        "agent_order": agent_order,
    }
    return model, meta


def ensure_env(
    env_name: str,
    ep_len: int,
    render_mode: Optional[str],
    agent_order: Optional[List[str]] = None,
    use_dynamics_env: bool = False,
    dynamics_ckpt: str = "",
    device: str = "auto",
):
    if use_dynamics_env or (isinstance(dynamics_ckpt, str) and dynamics_ckpt.strip()):
        if not dynamics_ckpt:
            raise ValueError("--dynamics_ckpt is required when using dynamics env")
        return DynamicsEnv(dynamics_ckpt, episode_length=ep_len, device=device)
    if env_name == "simple_tag_v3":
        kwargs = dict(max_cycles=ep_len, render_mode=render_mode)
        if agent_order:
            n_adv = sum(1 for a in agent_order if str(a).startswith("adversary_"))
            n_good = sum(1 for a in agent_order if str(a).startswith("agent_"))
            if n_adv > 0 and n_good > 0:
                kwargs["num_adversaries"] = int(n_adv)
                kwargs["num_good"] = int(n_good)
        try:
            env = simple_tag_v3.parallel_env(**kwargs)
        except TypeError:
            kwargs.pop("render_mode", None)
            env = simple_tag_v3.parallel_env(**kwargs)
        return env
    raise ValueError(f"Unsupported env_name: {env_name}")


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, infos = out
    else:
        obs, infos = out, {}
    return obs, infos


def step_env(env, actions: Dict[str, int]):
    out = env.step(actions)
    if isinstance(out, tuple) and len(out) == 5:
        obs, rewards, terms, truncs, infos = out
        dones = {a: bool(terms.get(a, False) or truncs.get(a, False)) for a in rewards.keys()}
        return obs, rewards, dones, infos
    if isinstance(out, tuple) and len(out) == 4:
        obs, rewards, dones, infos = out
        dones = {a: bool(dones.get(a, False)) for a in rewards.keys()}
        return obs, rewards, dones, infos
    raise ValueError("Unrecognized env.step return format.")


def obs_to_vec(obs_any: Any, obs_dim_expected: int) -> np.ndarray:
    if obs_any is None:
        return np.zeros((obs_dim_expected,), dtype=np.float32)
    arr = np.asarray(obs_any, dtype=np.float32).reshape(-1)
    if arr.shape[0] == obs_dim_expected:
        return arr
    if arr.shape[0] > obs_dim_expected:
        return arr[:obs_dim_expected]
    out = np.zeros((obs_dim_expected,), dtype=np.float32)
    out[:arr.shape[0]] = arr
    return out


@torch.no_grad()
def policy_action(model: BCPolicy,
                  obs_vec: np.ndarray,
                  agent_id: int,
                  sample: bool,
                  temperature: float,
                  device: torch.device) -> int:
    obs_t = torch.from_numpy(obs_vec.astype(np.float32)).unsqueeze(0).to(device)
    if model.use_agent_id:
        aid_t = torch.tensor([agent_id], dtype=torch.long, device=device)
        logits = model(obs_t, aid_t)
    else:
        logits = model(obs_t, None)

    if sample:
        t = max(1e-6, float(temperature))
        probs = torch.softmax(logits / t, dim=-1).squeeze(0)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def expert_vote_and_logp(expert: BCPolicy,
                         obs_vec: np.ndarray,
                         agent_id: int,
                         action_id: int,
                         device: torch.device) -> Tuple[int, float]:
    obs_t = torch.from_numpy(obs_vec.astype(np.float32)).unsqueeze(0).to(device)
    if expert.use_agent_id:
        aid_t = torch.tensor([agent_id], dtype=torch.long, device=device)
        logits = expert(obs_t, aid_t)
    else:
        logits = expert(obs_t, None)

    logp = F.log_softmax(logits, dim=-1)[0, action_id].item()
    argmax_a = int(torch.argmax(logits, dim=-1).item())
    return argmax_a, float(logp)


def parse_expert_ckpts(s: str) -> List[str]:
    paths = [x.strip() for x in s.split(",") if x.strip()]
    if len(paths) == 0:
        raise ValueError("--expert_ckpts parsed empty. Make sure it's a comma-separated string.")
    return paths


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--env_name", type=str, default="simple_tag_v3")
    p.add_argument("--episodes", type=int, required=True, help="number of KEPT (accepted) episodes")
    p.add_argument("--keep_steps", type=int, default=25, help="must reach this many steps to accept an episode")
    p.add_argument("--ep_len", type=int, default=25, help="passed to env max_cycles")
    p.add_argument("--render_mode", type=str, default=None)
    p.add_argument("--use_dynamics_env", action="store_true")
    p.add_argument("--dynamics_ckpt", type=str, default="")

    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--expert_ckpts", type=str, required=True, help="comma-separated list of ckpt paths")
    p.add_argument("--out_csv", type=str, required=True)

    p.add_argument("--min_vote", type=float, default=0.6)
    p.add_argument("--min_avg_logp", type=float, default=-2.0)

    p.add_argument("--student_sample", action="store_true", help="if set: sample from softmax; else argmax")
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--max_attempts", type=int, default=200000,
                   help="max rollout attempts to collect the required kept episodes (avoid infinite loop)")

    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    student, s_meta = load_bc(args.student_ckpt, device=device)
    expert_paths = parse_expert_ckpts(args.expert_ckpts)

    experts = []
    e_metas = []
    for path in expert_paths:
        m, meta = load_bc(path, device=device)
        experts.append(m)
        e_metas.append(meta)

    obs_dim = s_meta["obs_dim"]
    action_dim = s_meta["action_dim"]
    for meta in e_metas:
        if meta["obs_dim"] != obs_dim or meta["action_dim"] != action_dim:
            raise ValueError("Expert ckpt obs_dim/action_dim mismatch with student ckpt.")

    agent_order: List[str] = list(s_meta["agent_order"])
    agent_to_id = {a: i for i, a in enumerate(agent_order)}
    env = ensure_env(
        args.env_name,
        ep_len=args.ep_len,
        render_mode=args.render_mode,
        agent_order=agent_order,
        use_dynamics_env=args.use_dynamics_env,
        dynamics_ckpt=args.dynamics_ckpt,
        device=args.device,
    )

    rows: List[Dict[str, Any]] = []
    kept_episodes = 0
    attempt_episodes = 0

    total_kept_rows = 0
    total_discard_reject = 0
    total_discard_short = 0

    while kept_episodes < args.episodes and attempt_episodes < args.max_attempts:
        attempt_episodes += 1
        obs_dict, _ = reset_env(env)

        episode_agents = list(agent_order)

        rows_ep: List[Dict[str, Any]] = []
        episode_ok = True

        for t in range(args.keep_steps):
            if obs_dict is None or not isinstance(obs_dict, dict):
                episode_ok = False
                break
            if any(a not in obs_dict for a in episode_agents):
                episode_ok = False
                break

            actions: Dict[str, int] = {}
            per_agent_cache: Dict[str, Dict[str, Any]] = {}

            for agent in episode_agents:
                obs_vec = obs_to_vec(obs_dict[agent], obs_dim)
                agent_id = agent_to_id.get(agent, 0)

                a_student = policy_action(
                    student, obs_vec, agent_id,
                    sample=args.student_sample,
                    temperature=args.temperature,
                    device=device
                )

                votes = 0
                logps = []
                for ex in experts:
                    ex_argmax, ex_logp = expert_vote_and_logp(ex, obs_vec, agent_id, a_student, device=device)
                    if ex_argmax == a_student:
                        votes += 1
                    logps.append(ex_logp)

                vote_ratio = votes / max(1, len(experts))
                avg_logp = float(np.mean(logps)) if len(logps) > 0 else -1e9
                keep = (vote_ratio >= args.min_vote) and (avg_logp >= args.min_avg_logp)

                actions[agent] = int(a_student)
                per_agent_cache[agent] = {
                    "obs_vec": obs_vec,
                    "action_id": int(a_student),
                    "vote_ratio": float(vote_ratio),
                    "avg_logp": float(avg_logp),
                    "keep": bool(keep),
                }

            if any(not x["keep"] for x in per_agent_cache.values()):
                total_discard_reject += 1
                episode_ok = False
                break

            next_obs, rewards, dones, infos = step_env(env, actions)

            for agent in episode_agents:
                nobs_vec = obs_to_vec(next_obs.get(agent, None) if isinstance(next_obs, dict) else None, obs_dim)
                r = float(rewards.get(agent, 0.0)) if isinstance(rewards, dict) else 0.0
                d = bool(dones.get(agent, False)) if isinstance(dones, dict) else False

                row = {
                    "episode": int(kept_episodes),
                    "t": int(t),
                    "agent": agent,
                    "action_id": int(per_agent_cache[agent]["action_id"]),
                    "reward": r,
                    "done": int(d),
                    "vote_ratio": float(per_agent_cache[agent]["vote_ratio"]),
                    "avg_logp": float(per_agent_cache[agent]["avg_logp"]),
                }
                for i in range(obs_dim):
                    row[f"obs_{i}"] = float(per_agent_cache[agent]["obs_vec"][i])
                for i in range(obs_dim):
                    row[f"next_obs_{i}"] = float(nobs_vec[i])

                rows_ep.append(row)

            obs_dict = next_obs

            if isinstance(dones, dict) and len(dones) > 0 and all(bool(x) for x in dones.values()):
                if t != args.keep_steps - 1:
                    total_discard_short += 1
                    episode_ok = False
                break

        if episode_ok:
            expected_rows = args.keep_steps * len(agent_order)
            if len(rows_ep) == expected_rows:
                rows.extend(rows_ep)
                total_kept_rows += len(rows_ep)
                kept_episodes += 1
                if kept_episodes % 10 == 0 or kept_episodes == 1:
                    print(f"[kept {kept_episodes}/{args.episodes}] attempts={attempt_episodes} ")
            else:
                total_discard_short += 1
        else:
            pass

    if kept_episodes < args.episodes:
        print(f"WARNING: only kept {kept_episodes}/{args.episodes} episodes within max_attempts={args.max_attempts}. "
              f"Try relaxing gate (min_vote/min_avg_logp) or increase max_attempts.")

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    print(f"Saved CSV: {args.out_csv}")
    print(f"Summary: kept_episodes={kept_episodes}, attempts={attempt_episodes}, "
          f"kept_rows={total_kept_rows}, discard_reject_eps={total_discard_reject}, discard_short_eps={total_discard_short}, "
          f"gate(min_vote={args.min_vote}, min_avg_logp={args.min_avg_logp})")


if __name__ == "__main__":
    main()
