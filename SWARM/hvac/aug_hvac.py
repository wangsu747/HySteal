from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from HVAC.config import load_config
from HVAC.env import HVACCampusEnv
from SWARM.aug_env import BCPolicy, expert_vote_and_logp, load_bc, policy_action
from SWARM.dyn_csv import DynCfg, ResidualDynamicsMLP
from SWARM.utils.il_csv_adapter import normalize_il_dataframe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Gated HVAC rollout augmentation using learned dynamics.")
    p.add_argument("--config", type=str, default="HVAC/configs/campus_4agent.yml")
    p.add_argument("--start_csv", type=str, required=True, help="Expert IL CSV used to sample initial states.")
    p.add_argument(
        "--start_mode",
        type=str,
        default="t0",
        choices=["t0", "any"],
        help="Sample augmentation starts only from episode starts (t0) or from any full joint state in the expert CSV.",
    )
    p.add_argument("--dynamics_ckpt", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--expert_ckpts", type=str, required=True, help="comma-separated list")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--episodes", type=int, required=True, help="number of kept episodes")
    p.add_argument("--keep_steps", type=int, default=20)
    p.add_argument("--min_vote", type=float, default=0.6)
    p.add_argument("--min_avg_logp", type=float, default=-2.0)
    p.add_argument("--student_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_attempts", type=int, default=200000)
    p.add_argument(
        "--start_noise_sigma",
        type=float,
        default=0.0,
        help="Gaussian noise applied once to the sampled start joint-state (normalized obs space).",
    )
    p.add_argument(
        "--rollout_noise_sigma",
        type=float,
        default=0.0,
        help="Gaussian noise applied at every learned-dynamics step before student/expert action selection.",
    )
    p.add_argument(
        "--noise_dims",
        type=str,
        default="0-5,7-8,16-18",
        help="Comma-separated normalized observation dims to perturb, e.g. '0-5,7-8,16-18'.",
    )
    p.add_argument("--save_meta_json", type=str, default="")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def parse_expert_ckpts(s: str) -> List[str]:
    paths = [x.strip() for x in s.split(",") if x.strip()]
    if not paths:
        raise ValueError("--expert_ckpts parsed empty.")
    return paths


def load_dynamics(ckpt_path: str, device: torch.device) -> Tuple[ResidualDynamicsMLP, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ResidualDynamicsMLP(
        DynCfg(
            in_dim=int(ckpt["joint_state_dim"]) + int(ckpt["joint_action_dim"]),
            out_dim=int(ckpt["joint_state_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            n_layers=int(ckpt["n_layers"]),
        )
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_start_pool(
    csv_path: str,
    agent_order: List[str],
    obs_dim: int,
    start_mode: str,
) -> List[np.ndarray]:
    df = pd.read_csv(csv_path, low_memory=False)
    df, _ = normalize_il_dataframe(df, obs_dim=obs_dim, fill_value=0.0, ensure_next_obs=True)
    keep_cols = ["episode", "t", "agent"] + [f"obs_{i}" for i in range(obs_dim)]
    for c in keep_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in start_csv: {c}")
    d = df[keep_cols].copy()
    d = d[d["agent"].isin(agent_order)].copy()
    agent_to_idx = {a: i for i, a in enumerate(agent_order)}

    starts: List[np.ndarray] = []
    for (_ep, t), g in d.groupby(["episode", "t"], sort=True):
        if start_mode == "t0" and int(t) != 0:
            continue
        if len(g) != len(agent_order):
            continue
        if g["agent"].nunique() != len(agent_order):
            continue
        g = g.sort_values(by="agent", key=lambda s: s.map(agent_to_idx))
        if list(g["agent"].values) != agent_order:
            continue
        obs = g[[f"obs_{i}" for i in range(obs_dim)]].to_numpy(dtype=np.float32)
        starts.append(obs.reshape(-1))
    if not starts:
        raise RuntimeError(f"No valid full-joint states found in start_csv for start_mode={start_mode}.")
    return starts


def parse_noise_dims(spec: str, obs_dim: int) -> np.ndarray:
    mask = np.zeros((obs_dim,), dtype=bool)
    if not isinstance(spec, str) or not spec.strip():
        return mask
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = max(0, int(lo_s))
            hi = min(obs_dim - 1, int(hi_s))
            if lo <= hi:
                mask[lo:hi + 1] = True
        else:
            idx = int(part)
            if 0 <= idx < obs_dim:
                mask[idx] = True
    return mask


def apply_obs_noise(
    flat_state: np.ndarray,
    num_agents: int,
    obs_dim: int,
    sigma: float,
    noise_mask: np.ndarray,
) -> np.ndarray:
    if sigma <= 0.0 or not np.any(noise_mask):
        return flat_state
    arr = np.asarray(flat_state, dtype=np.float32).reshape(num_agents, obs_dim).copy()
    noise = np.random.randn(num_agents, obs_dim).astype(np.float32) * float(sigma)
    noise[:, ~noise_mask] = 0.0
    arr += noise
    arr = np.clip(arr, -2.0, 2.0)
    if obs_dim >= 23:
        arr[:, 19:23] = (arr[:, 19:23] > 0.5).astype(np.float32)
    return arr.reshape(-1)


def flat_to_agent_obs(env: HVACCampusEnv, flat_state: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(flat_state, dtype=np.float32).reshape(env.num_agents, env.obs_dim)
    return {
        agent_id: arr[idx].copy()
        for idx, agent_id in enumerate(env.agent_ids)
    }


def joint_action_onehot(env: HVACCampusEnv, action_dict: Dict[str, int]) -> np.ndarray:
    out = np.zeros((env.num_agents, env.num_actions), dtype=np.float32)
    for idx, agent_id in enumerate(env.agent_ids):
        aid = int(action_dict[agent_id])
        if 0 <= aid < env.num_actions:
            out[idx, aid] = 1.0
    return out.reshape(-1)


def infer_reward_from_predicted_state(
    env: HVACCampusEnv,
    prev_obs_dict: Dict[str, np.ndarray],
    next_obs_dict: Dict[str, np.ndarray],
    action_dict: Dict[str, int],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    total_power_ratio = float(np.mean([next_obs_dict[a][16] for a in env.agent_ids]))
    total_power_kw = total_power_ratio * env.config.campus_power_budget_kw

    peak_ratio = float(np.mean([next_obs_dict[a][17] for a in env.agent_ids]))
    peak_norm = max(25.0, 0.25 * env.config.campus_power_budget_kw)
    peak_excess_kw = peak_ratio * peak_norm

    curtailment_ratio = float(np.mean([next_obs_dict[a][18] for a in env.agent_ids]))
    price = float(np.mean([prev_obs_dict[a][14] for a in env.agent_ids]))
    energy_term = env._energy_penalty(total_power_kw)
    peak_term = env._peak_penalty(peak_excess_kw)
    price_term = env.config.cost_weight * price * (total_power_kw / 1000.0)
    team_reward = -(energy_term + peak_term + price_term)
    total_requested_power_kw = (
        float(total_power_kw / max(1e-6, 1.0 - curtailment_ratio))
        if curtailment_ratio < 0.999
        else float(total_power_kw)
    )

    rewards: Dict[str, float] = {}
    info_agents: Dict[str, Dict[str, float]] = {}
    for idx, agent_id in enumerate(env.agent_ids):
        prev_obs = prev_obs_dict[agent_id]
        next_obs = next_obs_dict[agent_id]
        last_action = float(prev_obs[6])
        raw_action = float(env.action_bins[int(action_dict[agent_id])])
        action_penalty = env.config.action_smooth_weight * abs(raw_action - last_action)
        occupancy = float(next_obs[4])
        comfort_violation = float(next_obs[7]) * 5.0
        comfort_penalty = env._comfort_penalty(comfort_violation, occupancy)
        allocated_power_kw = float(next_obs[5]) * 500.0
        allocated_fraction = float(next_obs[8])
        requested_power_kw = (
            float(allocated_power_kw / max(1e-6, allocated_fraction))
            if allocated_fraction > 1e-6
            else float(allocated_power_kw)
        )
        rewards[agent_id] = team_reward - comfort_penalty - action_penalty
        info_agents[agent_id] = {
            "team_reward": float(team_reward),
            "energy_penalty": float(energy_term),
            "peak_penalty": float(peak_term),
            "comfort_penalty": float(comfort_penalty),
            "action_penalty": float(action_penalty),
            "action_idx": int(action_dict[agent_id]),
            "raw_action": raw_action,
            "occupancy": occupancy,
            "requested_power_kw": requested_power_kw,
            "allocated_power_kw": allocated_power_kw,
            "allocated_fraction": allocated_fraction,
            "urgency_score": float("nan"),
            "power_kw": allocated_power_kw,
            "indoor_temp_c": float(next_obs[0]) * 40.0,
            "comfort_violation": comfort_violation,
        }
    return rewards, {
        "total_power_kw": float(total_power_kw),
        "total_requested_power_kw": total_requested_power_kw,
        "peak_excess_kw": float(peak_excess_kw),
        "curtailment_ratio": float(curtailment_ratio),
        "price": float(price),
        "agents": info_agents,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = load_config(args.config)
    env = HVACCampusEnv(cfg.env, seed=args.seed)

    student, s_meta = load_bc(args.student_ckpt, device=device)
    expert_paths = parse_expert_ckpts(args.expert_ckpts)
    experts = []
    expert_metas = []
    for path in expert_paths:
        m, meta = load_bc(path, device=device)
        experts.append(m)
        expert_metas.append(meta)

    obs_dim = int(s_meta["obs_dim"])
    action_dim = int(s_meta["action_dim"])
    agent_order = list(s_meta["agent_order"])
    if agent_order != env.agent_ids:
        raise ValueError(f"Student checkpoint agent_order={agent_order} does not match HVAC env agent_ids={env.agent_ids}")
    for meta in expert_metas:
        if int(meta["obs_dim"]) != obs_dim or int(meta["action_dim"]) != action_dim:
            raise ValueError("Expert ckpt obs_dim/action_dim mismatch with student ckpt.")
        if list(meta["agent_order"]) != agent_order:
            raise ValueError("Expert ckpt agent_order mismatch with student ckpt.")

    dyn_model, dyn_meta = load_dynamics(args.dynamics_ckpt, device=device)
    if int(dyn_meta["obs_dim_per_agent"]) != obs_dim:
        raise ValueError("Dynamics obs_dim_per_agent mismatch with BC checkpoints.")
    if int(dyn_meta["action_n"]) != action_dim:
        raise ValueError("Dynamics action_n mismatch with BC checkpoints.")
    if list(dyn_meta["agent_order"]) != agent_order:
        raise ValueError("Dynamics agent_order mismatch with BC checkpoints.")

    starts = load_start_pool(
        args.start_csv,
        agent_order=agent_order,
        obs_dim=obs_dim,
        start_mode=args.start_mode,
    )
    agent_to_id = {a: i for i, a in enumerate(agent_order)}
    noise_mask = parse_noise_dims(args.noise_dims, obs_dim)

    rows: List[Dict[str, Any]] = []
    kept_episodes = 0
    attempt_episodes = 0
    total_kept_rows = 0
    total_discard_reject = 0

    while kept_episodes < args.episodes and attempt_episodes < args.max_attempts:
        attempt_episodes += 1
        current_flat = starts[random.randrange(len(starts))].copy()
        current_flat = apply_obs_noise(
            current_flat,
            num_agents=env.num_agents,
            obs_dim=obs_dim,
            sigma=float(args.start_noise_sigma),
            noise_mask=noise_mask,
        )
        rows_ep: List[Dict[str, Any]] = []
        episode_ok = True

        for t in range(args.keep_steps):
            current_flat = apply_obs_noise(
                current_flat,
                num_agents=env.num_agents,
                obs_dim=obs_dim,
                sigma=float(args.rollout_noise_sigma),
                noise_mask=noise_mask,
            )
            obs_dict = flat_to_agent_obs(env, current_flat)
            actions: Dict[str, int] = {}
            per_agent_cache: Dict[str, Dict[str, Any]] = {}

            for agent in agent_order:
                obs_vec = np.asarray(obs_dict[agent], dtype=np.float32)
                agent_id = agent_to_id[agent]
                a_student = policy_action(
                    student,
                    obs_vec,
                    agent_id,
                    sample=args.student_sample,
                    temperature=args.temperature,
                    device=device,
                )
                votes = 0
                logps = []
                for ex in experts:
                    ex_argmax, ex_logp = expert_vote_and_logp(ex, obs_vec, agent_id, a_student, device=device)
                    if ex_argmax == a_student:
                        votes += 1
                    logps.append(ex_logp)
                vote_ratio = votes / max(1, len(experts))
                avg_logp = float(np.mean(logps)) if logps else -1e9
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

            s_t = torch.from_numpy(current_flat).unsqueeze(0).to(device)
            a_t = torch.from_numpy(joint_action_onehot(env, actions)).unsqueeze(0).to(device)
            with torch.no_grad():
                next_flat_t, _ = dyn_model(s_t, a_t)
            next_flat = next_flat_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            next_obs_dict = flat_to_agent_obs(env, next_flat)
            rewards, info = infer_reward_from_predicted_state(env, obs_dict, next_obs_dict, actions)
            done = int(t == args.keep_steps - 1)

            for agent in agent_order:
                row = {
                    "episode": int(kept_episodes),
                    "t": int(t),
                    "agent": agent,
                    "action_id": int(per_agent_cache[agent]["action_id"]),
                    "reward": float(rewards[agent]),
                    "done": int(done),
                    "vote_ratio": float(per_agent_cache[agent]["vote_ratio"]),
                    "avg_logp": float(per_agent_cache[agent]["avg_logp"]),
                    "obs_json": json.dumps(per_agent_cache[agent]["obs_vec"].tolist()),
                    "next_obs_json": json.dumps(np.asarray(next_obs_dict[agent], dtype=np.float32).tolist()),
                }
                for i in range(obs_dim):
                    row[f"obs_{i}"] = float(per_agent_cache[agent]["obs_vec"][i])
                for i in range(obs_dim):
                    row[f"next_obs_{i}"] = float(next_obs_dict[agent][i])
                rows_ep.append(row)

            current_flat = next_flat

        if episode_ok:
            expected_rows = args.keep_steps * len(agent_order)
            if len(rows_ep) == expected_rows:
                rows.extend(rows_ep)
                total_kept_rows += len(rows_ep)
                kept_episodes += 1
                if kept_episodes % 10 == 0 or kept_episodes == 1:
                    print(f"[kept {kept_episodes}/{args.episodes}] attempts={attempt_episodes} rows={total_kept_rows}")

    if kept_episodes < args.episodes:
        print(
            f"WARNING: only kept {kept_episodes}/{args.episodes} episodes within max_attempts={args.max_attempts}. "
            f"Try relaxing gate or increasing max_attempts."
        )

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved CSV: {args.out_csv}")
    print(
        f"Summary: kept_episodes={kept_episodes}, attempts={attempt_episodes}, "
        f"kept_rows={total_kept_rows}, discard_reject_eps={total_discard_reject}, "
        f"gate(min_vote={args.min_vote}, min_avg_logp={args.min_avg_logp})"
    )

    if isinstance(args.save_meta_json, str) and args.save_meta_json.strip():
        meta = {
            "config": str(Path(args.config).resolve()),
            "start_csv": str(Path(args.start_csv).resolve()),
            "dynamics_ckpt": str(Path(args.dynamics_ckpt).resolve()),
            "student_ckpt": str(Path(args.student_ckpt).resolve()),
            "expert_ckpts": [str(Path(x).resolve()) for x in expert_paths],
            "episodes": int(args.episodes),
            "keep_steps": int(args.keep_steps),
            "start_mode": str(args.start_mode),
            "min_vote": float(args.min_vote),
            "min_avg_logp": float(args.min_avg_logp),
            "student_sample": bool(args.student_sample),
            "temperature": float(args.temperature),
            "start_noise_sigma": float(args.start_noise_sigma),
            "rollout_noise_sigma": float(args.rollout_noise_sigma),
            "noise_dims": str(args.noise_dims),
            "kept_episodes": int(kept_episodes),
            "attempt_episodes": int(attempt_episodes),
            "kept_rows": int(total_kept_rows),
        }
        meta_path = Path(args.save_meta_json).resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved meta JSON: {meta_path}")


if __name__ == "__main__":
    main()
