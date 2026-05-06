#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.aug_env import (
    expert_vote_and_logp,
    get_device,
    load_bc,
    obs_to_vec,
    policy_action,
    set_seed,
)
from SWARM.lenv import LearnedDynamicsParallelEnv


def main():
    p = argparse.ArgumentParser("Generate vote-filtered augmentation rollouts inside a learned dynamics environment")

    p.add_argument("--episodes", type=int, required=True, help="number of kept episodes")
    p.add_argument("--keep_steps", type=int, default=25)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--expert_ckpts", type=str, required=True, help="comma-separated list of reviewer ckpts")
    p.add_argument("--dynamics_ckpt", type=str, required=True)
    p.add_argument("--init_csv_path", type=str, required=True, help="CSV used to build learned-env reset pool")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--save_meta_json", type=str, default="")

    p.add_argument("--min_vote", type=float, default=0.5)
    p.add_argument("--min_avg_logp", type=float, default=-1.8)
    p.add_argument("--student_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_attempts", type=int, default=200000)
    p.add_argument("--reset_from_t0_only", action="store_true")
    p.add_argument("--reset_noise_sigma", type=float, default=0.0)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    student, s_meta = load_bc(args.student_ckpt, device=device)
    expert_paths = [x.strip() for x in args.expert_ckpts.split(",") if x.strip()]
    if not expert_paths:
        raise ValueError("--expert_ckpts parsed empty.")
    experts = []
    e_metas = []
    for path in expert_paths:
        m, meta = load_bc(path, device=device)
        experts.append(m)
        e_metas.append(meta)

    obs_dim = s_meta["obs_dim"]
    action_dim = s_meta["action_dim"]
    agent_order = list(s_meta["agent_order"])
    agent_to_id = {a: i for i, a in enumerate(agent_order)}
    for meta in e_metas:
        if meta["obs_dim"] != obs_dim or meta["action_dim"] != action_dim or list(meta["agent_order"]) != agent_order:
            raise ValueError("Expert ckpt layout mismatch with student ckpt.")

    env = LearnedDynamicsParallelEnv(
        dynamics_ckpt_path=args.dynamics_ckpt,
        init_csv_path=args.init_csv_path,
        max_cycles=args.keep_steps,
        reset_from_t0_only=bool(args.reset_from_t0_only),
        reset_noise_sigma=float(args.reset_noise_sigma),
        device=str(device),
    )

    rows: List[Dict[str, Any]] = []
    kept_episodes = 0
    attempt_episodes = 0
    total_discard_reject = 0
    total_discard_short = 0

    while kept_episodes < args.episodes and attempt_episodes < args.max_attempts:
        attempt_episodes += 1
        obs_dict, _ = env.reset(seed=args.seed + attempt_episodes)
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

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            dones = {a: bool(terms.get(a, False) or truncs.get(a, False)) for a in episode_agents}

            for agent in episode_agents:
                nobs_vec = obs_to_vec(next_obs.get(agent, None) if isinstance(next_obs, dict) else None, obs_dim)
                r = float(rewards.get(agent, 0.0)) if isinstance(rewards, dict) else 0.0
                d = bool(dones.get(agent, False))
                row = {
                    "episode": int(kept_episodes),
                    "t": int(t),
                    "agent": agent,
                    "action_id": int(per_agent_cache[agent]["action_id"]),
                    "action": int(per_agent_cache[agent]["action_id"]),
                    "reward": r,
                    "done": int(d),
                    "vote_ratio": float(per_agent_cache[agent]["vote_ratio"]),
                    "avg_logp": float(per_agent_cache[agent]["avg_logp"]),
                    "source": "aug",
                    "weight": 1.0,
                    "obs_json": json.dumps(per_agent_cache[agent]["obs_vec"].astype(np.float32).tolist()),
                    "next_obs_json": json.dumps(nobs_vec.astype(np.float32).tolist()),
                }
                for i in range(obs_dim):
                    row[f"obs_{i}"] = float(per_agent_cache[agent]["obs_vec"][i])
                    row[f"next_obs_{i}"] = float(nobs_vec[i])
                rows_ep.append(row)

            obs_dict = next_obs
            if all(bool(dones.get(a, False)) for a in episode_agents):
                if t != args.keep_steps - 1:
                    total_discard_short += 1
                    episode_ok = False
                break

        if episode_ok:
            expected_rows = args.keep_steps * len(agent_order)
            if len(rows_ep) == expected_rows:
                rows.extend(rows_ep)
                kept_episodes += 1
                if kept_episodes % 10 == 0 or kept_episodes == 1:
                    print(f"[kept {kept_episodes}/{args.episodes}] attempts={attempt_episodes}")
            else:
                total_discard_short += 1

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved CSV: {args.out_csv}")
    print(
        f"Summary: kept_episodes={kept_episodes}, attempts={attempt_episodes}, "
        f"discard_reject_eps={total_discard_reject}, discard_short_eps={total_discard_short}, "
        f"gate(min_vote={args.min_vote}, min_avg_logp={args.min_avg_logp})"
    )

    if args.save_meta_json:
        meta = {
            "episodes_requested": int(args.episodes),
            "episodes_kept": int(kept_episodes),
            "attempt_episodes": int(attempt_episodes),
            "keep_steps": int(args.keep_steps),
            "student_ckpt": args.student_ckpt,
            "expert_ckpts": expert_paths,
            "dynamics_ckpt": args.dynamics_ckpt,
            "init_csv_path": args.init_csv_path,
            "agent_order": agent_order,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "min_vote": float(args.min_vote),
            "min_avg_logp": float(args.min_avg_logp),
            "reset_from_t0_only": bool(args.reset_from_t0_only),
            "reset_noise_sigma": float(args.reset_noise_sigma),
        }
        with open(args.save_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved meta: {args.save_meta_json}")


if __name__ == "__main__":
    main()
