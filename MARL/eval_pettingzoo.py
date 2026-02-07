











import argparse
import json
import os
import copy
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from main import get_env
from MADDPG import MADDPG

DEFAULT_AGENT_ORDER = ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']


def parse_reset(reset_out):




    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, infos = reset_out
        return obs, infos
    return reset_out, {}


def ensure_obs_dict(obs, agents) -> Dict[str, np.ndarray]:

    if isinstance(obs, dict):
        return obs
    if isinstance(obs, (list, tuple)):
        return {a: ob for a, ob in zip(agents, obs)}
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def done_all(terminations: Dict, truncations: Dict, agents: List[str]) -> bool:
    if not agents:
        return True
    return all(bool(terminations.get(a, False) or truncations.get(a, False)) for a in agents)


def try_get_world(env):

    candidates = [
        getattr(env, "unwrapped", None),
        getattr(getattr(env, "aec_env", None), "unwrapped", None),
        getattr(getattr(env, "env", None), "unwrapped", None),
        getattr(getattr(getattr(env, "env", None), "env", None), "unwrapped", None),
    ]
    for base in candidates:
        if base is None:
            continue
        w = getattr(base, "world", None)
        if w is not None:
            return w
    return None


def snapshot_world(env):

    world = try_get_world(env)
    if world is None:
        return None
    return copy.deepcopy(world)


def restore_world(env, world_snapshot):

    if world_snapshot is None:
        return
    world = try_get_world(env)
    if world is None:
        return
    world.__dict__.clear()
    world.__dict__.update(copy.deepcopy(world_snapshot.__dict__))


def get_landmarks_xy(env) -> List[Tuple[float, float]]:

    world = try_get_world(env)
    if world is None:
        return []
    landmarks = getattr(world, "landmarks", None)
    if not landmarks:
        return []
    out = []
    for lm in landmarks:
        st = getattr(lm, "state", None)
        p = getattr(st, "p_pos", None) if st is not None else None
        if p is None:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def get_agent_xy(obs_vec: np.ndarray) -> Tuple[Optional[float], Optional[float]]:




    if obs_vec is None:
        return None, None
    if len(obs_vec) >= 4:
        return float(obs_vec[2]), float(obs_vec[3])
    return None, None


def normalize_action_to_id_and_json(a) -> Tuple[int, str]:





    if a is None:
        return 0, ""

    if isinstance(a, (np.integer,)):
        return int(a), ""

    if isinstance(a, int):
        return int(a), ""

    if isinstance(a, float):
        return int(a), ""

    if isinstance(a, (list, tuple, np.ndarray)):
        arr = np.array(a)
        if arr.ndim == 0:
            return int(arr.item()), ""
        a_id = int(np.argmax(arr))
        return a_id, json.dumps(arr.tolist())

    return 0, json.dumps(str(a))


def pad_or_truncate(vec: np.ndarray, target_dim: int) -> np.ndarray:

    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[0] == target_dim:
        return vec
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    out = np.full((target_dim,), np.nan, dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="simple_tag_v3",
                        choices=["simple_adversary_v3", "simple_spread_v3", "simple_tag_v3"])
    parser.add_argument("--folder", type=str, required=True,
                        help="results/<env_name>/<folder>/  model.pt  folder 1, 91")

    parser.add_argument("--episode_num", type=int, default=1000, help="test episodes")
    parser.add_argument("--keep_steps", type=int, default=50, help=" max_cycles")

    parser.add_argument("--start_mode", type=str, default="random", choices=["random", "fixed"],
                        help="Episode start mode: random=env reset random init, fixed=same initial world each episode")

    parser.add_argument("--out_positions", type=str, default="eval_positions_1000_50_random.csv",
                        help="CSVlandmark")
    parser.add_argument("--out_il", type=str, default="il_transitions_1000_50_random.csv",
                        help="IL(s,a,s') CSVagent")

    parser.add_argument("--render_mode", type=str, default=None,
                        choices=[None, "rgb_array"], nargs="?",
                        help=" rgb_arrayrender")

    args = parser.parse_args()

    model_dir = os.path.join("./results", args.env_name, str(args.folder))
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model.pt: {model_path}")

    out_dir = os.path.join(model_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    positions_path = os.path.join(out_dir, args.out_positions)
    il_path = os.path.join(out_dir, args.out_il)

    env, dim_info = get_env(args.env_name, ep_len=args.keep_steps, render_mode=args.render_mode)

    maddpg = MADDPG.load(dim_info, model_path)

    env_agents = list(env.agents)
    agent_order = DEFAULT_AGENT_ORDER if all(a in env_agents for a in DEFAULT_AGENT_ORDER) else env_agents

    positions_rows: List[List] = []
    il_rows: List[Dict] = []

    obs_dim: Optional[int] = None

    lm_xy_static = get_landmarks_xy(env)
    lm_cols = []
    for i in range(len(lm_xy_static)):
        lm_cols += [f"lm{i}_x", f"lm{i}_y"]

    world_snapshot = None

    for ep in range(args.episode_num):

        obs, infos = parse_reset(env.reset())
        obs = ensure_obs_dict(obs, env.agents)

        if args.start_mode == "fixed":
            if world_snapshot is None:
                world_snapshot = snapshot_world(env)
                if world_snapshot is None:
                    print("[warn] start_mode=fixed but cannot access env.world (wrappers?). fallback to random.")
            else:
                restore_world(env, world_snapshot)
                obs, infos = parse_reset(env.reset())
                obs = ensure_obs_dict(obs, env.agents)

        if obs_dim is None:
            for a in agent_order:
                if a in obs and obs[a] is not None and len(obs[a]) > 0:
                    obs_dim = int(len(obs[a]))
                    break
            if obs_dim is None:
                raise RuntimeError("Cannot infer obs_dim from env.reset() output.")
            print(f"[info] inferred obs_dim = {obs_dim}")

        lm_xy = get_landmarks_xy(env)
        lm_flat = []
        for (lx, ly) in lm_xy:
            lm_flat.extend([lx, ly])

        for t in range(args.keep_steps):
            actions_raw = maddpg.select_action(obs)

            actions: Dict[str, int] = {}
            action_json_map: Dict[str, str] = {}
            for agent in agent_order:
                raw_a = actions_raw.get(agent, 0)
                a_id, a_json = normalize_action_to_id_and_json(raw_a)
                actions[agent] = int(a_id)
                action_json_map[agent] = a_json

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            next_obs = ensure_obs_dict(next_obs, env.agents)

            for agent in agent_order:
                if agent not in obs:
                    continue

                obs_vec = np.asarray(obs[agent], dtype=np.float32)
                nobs_vec = np.asarray(next_obs.get(agent, np.array([], dtype=np.float32)), dtype=np.float32)

                obs_vec_fix = pad_or_truncate(obs_vec, obs_dim)
                if nobs_vec.size == 0:
                    nobs_vec_fix = np.full((obs_dim,), np.nan, dtype=np.float32)
                else:
                    nobs_vec_fix = pad_or_truncate(nobs_vec, obs_dim)

                action_id = int(actions.get(agent, 0))
                action_json = action_json_map.get(agent, "")

                r = float(rewards.get(agent, 0.0))
                d = int(bool(terminations.get(agent, False) or truncations.get(agent, False)))

                x, y = get_agent_xy(obs_vec)
                positions_rows.append([ep, t, agent, x, y, action_id, *lm_flat])

                row = {
                    "episode": ep,
                    "t": t,
                    "agent": agent,
                    "action_id": action_id,
                    "action_json": action_json,
                    "reward": r,
                    "done": d,
                    "obs_json": json.dumps(obs_vec.tolist()),
                    "next_obs_json": json.dumps(nobs_vec.tolist()) if nobs_vec.size > 0 else "[]",
                }
                for i in range(obs_dim):
                    row[f"obs_{i}"] = float(obs_vec_fix[i]) if not np.isnan(obs_vec_fix[i]) else np.nan
                for i in range(obs_dim):
                    row[f"next_obs_{i}"] = float(nobs_vec_fix[i]) if not np.isnan(nobs_vec_fix[i]) else np.nan

                il_rows.append(row)

            obs = next_obs

            if done_all(terminations, truncations, list(env.agents)):
                break

        if (ep + 1) % 50 == 0:
            print(f"[eval] finished episode {ep+1}/{args.episode_num}")

    pos_cols = ["episode", "t", "agent", "x", "y", "action_id"] + lm_cols
    df_pos = pd.DataFrame(positions_rows, columns=pos_cols)
    df_pos.to_csv(positions_path, index=False)

    df_il = pd.DataFrame(il_rows)
    df_il.to_csv(il_path, index=False)

    env.close()

    print("[done] Saved:")
    print("  positions (for animation):", positions_path)
    print("  IL transitions (for BC/AP):", il_path)
    print("  start_mode:", args.start_mode)
    print("  IL columns example:", list(df_il.columns)[:20], "...")

    if obs_dim is not None:
        print(f"[info] obs_dim={obs_dim}. BC obs_0..obs_{obs_dim-1} -> action_id .")


if __name__ == "__main__":
    main()
