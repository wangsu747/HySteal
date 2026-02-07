


import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from MADDPG import MADDPG
from overcooked_wrapper import OvercookedParallelEnv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def ensure_dir(p: str):
    if p is None or p == "":
        return
    os.makedirs(p, exist_ok=True)


def load_policy(layout: str, folder: str, dim_info: Dict[str, List[int]]):
    model_dir = os.path.join("./results_overcooked", layout, str(folder))
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model.pt: {model_path}")
    maddpg = MADDPG.load(dim_info, model_path)
    return maddpg, model_dir, model_path


def try_make_gif_overcooked(
    frames,
    gif_path: str,
    fps: int = 8,
    grid_shape: Tuple[int, int] = (5, 5),
):
    if len(frames) == 0:
        return False

    H, W = int(grid_shape[0]), int(grid_shape[1])

    fig, ax = plt.subplots(figsize=(W * 0.8, H * 0.8))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True)

    p0_sc = ax.scatter([], [], s=180, marker="o")
    p1_sc = ax.scatter([], [], s=180, marker="o")

    p0_line, = ax.plot([], [], linewidth=2, alpha=0.6)
    p1_line, = ax.plot([], [], linewidth=2, alpha=0.6)

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def _empty_offsets():
        return np.zeros((0, 2), dtype=np.float32)

    def init():
        p0_sc.set_offsets(_empty_offsets())
        p1_sc.set_offsets(_empty_offsets())
        p0_line.set_data([], [])
        p1_line.set_data([], [])
        title.set_text("")
        return (p0_sc, p1_sc, p0_line, p1_line, title)

    def update(i):
        fr = frames[i]
        x0, y0 = fr["p0_xy"]
        x1, y1 = fr["p1_xy"]

        p0_sc.set_offsets(np.array([[x0, y0]], dtype=np.float32))
        p1_sc.set_offsets(np.array([[x1, y1]], dtype=np.float32))

        p0_hist = np.array([f["p0_xy"] for f in frames[:i+1]], dtype=np.float32)
        p1_hist = np.array([f["p1_xy"] for f in frames[:i+1]], dtype=np.float32)

        p0_line.set_data(p0_hist[:, 0], p0_hist[:, 1])
        p1_line.set_data(p1_hist[:, 0], p1_hist[:, 1])

        title.set_text(f"t={int(fr.get('t', i))}")
        return (p0_sc, p1_sc, p0_line, p1_line, title)

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True)
    ensure_dir(os.path.dirname(gif_path))
    anim.save(gif_path, writer=PillowWriter(fps=int(fps)))
    plt.close(fig)
    return True


def export_traj_csv_overcooked(traj_rows: List[List], out_dir: str, filename: str = "traj_entities.csv"):
    cols = ["episode", "t", "entity", "id", "x", "y", "dir", "held", "event"]
    df = pd.DataFrame(traj_rows, columns=cols)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print("[csv] saved traj:", out_path)


def export_mail_csv_head(episodes_obs, episodes_act, episodes_rew, episodes_done,
                         out_dir: str, filename: str = "mail_steps_head.csv",
                         max_episodes: int = 3, max_steps: int = 50):
    rows = []
    epN = min(len(episodes_obs), int(max_episodes))

    for ep in range(epN):
        obs_ep = episodes_obs[ep]
        act_ep = episodes_act[ep]
        rew_ep = episodes_rew[ep]
        done_ep = episodes_done[ep]
        T = min(obs_ep.shape[0], int(max_steps))
        N = obs_ep.shape[1]

        for t in range(T):
            for a in range(N):
                obs_head = obs_ep[t, a, :20].tolist()
                rows.append([ep, t, a, *obs_head, int(act_ep[t, a]), float(rew_ep[t, a]), bool(done_ep[t, a])])

    cols = ["episode", "t", "agent"] + [f"obs_{i}" for i in range(20)] + ["action", "reward", "done"]
    df = pd.DataFrame(rows, columns=cols)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print("[csv] saved mail head:", out_path)


def _player_info_from_state(state):

    players = getattr(state, "players", None)
    if players is None or len(players) < 2:
        raise RuntimeError("Cannot find state.players (need 2 players)")

    def _dir_to_int(o):
        try:
            if hasattr(o, "value"):
                return int(o.value)
            return int(o)
        except Exception:
            return -1

    def _held_to_int(h):
        if h is None:
            return 0
        name = getattr(h, "name", None)
        if name is None:
            name = getattr(h, "obj_type", None) or getattr(h, "type", None)
        s = str(name).lower()
        if "dish" in s:
            return 1
        if "onion" in s:
            return 2
        if "soup" in s:
            return 3
        if "tomato" in s:
            return 4
        return 9

    p0, p1 = players[0], players[1]
    x0, y0 = p0.position
    x1, y1 = p1.position
    d0 = _dir_to_int(getattr(p0, "orientation", -1))
    d1 = _dir_to_int(getattr(p1, "orientation", -1))
    h0 = _held_to_int(getattr(p0, "held_object", None))
    h1 = _held_to_int(getattr(p1, "held_object", None))
    return (int(x0), int(y0), d0, h0), (int(x1), int(y1), d1, h1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--layout", type=str, default="cramped_room",
                        choices=["cramped_room", "asymmetric_advantages", "coordination_ring"])
    parser.add_argument("--episode_length", type=int, default=400)

    parser.add_argument("--folder", type=str, required=True)

    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_positions", type=str, default="eval_positions_overcooked.csv",
                        help="eval.py-compatible positions CSV")
    parser.add_argument("--out_il", type=str, default="il_transitions_overcooked.csv",
                        help="eval.py-compatible IL transitions CSV")

    parser.add_argument("--out_mail", type=str, default="mail_dataset.npz")
    parser.add_argument("--out_traj", type=str, default="traj_dataset.npz")
    parser.add_argument("--save_meta", action="store_true")

    parser.add_argument("--shape_reward", action="store_true",
                        help="eval  shaped reward  IL reward")
    parser.add_argument("--team_reward", action="store_true",
                        help=" MAIL  reward  team  agent ")
    parser.add_argument("--quiet_planner", action="store_true",
                        help=" MotionPlanner  stdout ")

    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--gif_every", type=int, default=50)
    parser.add_argument("--gif_fps", type=int, default=8)

    parser.add_argument("--make_gif_all", action="store_true")
    parser.add_argument("--make_gif_goal", action="store_true",
                        help=" delivery episode  gif")

    parser.add_argument("--export_csv", action="store_true")
    parser.add_argument("--mail_csv_head_steps", type=int, default=50)

    parser.add_argument("--only_delivery_episodes", action="store_true",
                        help="MAIL  delivery  episodeTRAJ ")
    parser.add_argument("--delivery_last_k", type=int, default=0,
                        help=">0 delivery  K  delivery  MAIL")
    parser.add_argument("--min_delivery_step", type=int, default=0,
                        help=" delivery")

    args = parser.parse_args()
    np.random.seed(args.seed)

    feat_env = OvercookedParallelEnv(
        layout_name=args.layout,
        episode_length=args.episode_length,
        shape_reward=bool(args.shape_reward),
        render=False,
        quiet=bool(args.quiet_planner) if hasattr(OvercookedParallelEnv, "__init__") else False,
    )

    n_agents = feat_env.n_agents
    obs_dim = feat_env.obs_dim
    act_dim = feat_env.act_dim
    dim_info = {f"agent_{i}": [obs_dim, act_dim] for i in range(n_agents)}

    maddpg, model_dir, model_path = load_policy(args.layout, args.folder, dim_info)

    out_dir = os.path.join(model_dir, "eval")
    ensure_dir(out_dir)
    mail_path = os.path.join(out_dir, args.out_mail)
    traj_path = os.path.join(out_dir, args.out_traj)
    positions_path = os.path.join(out_dir, args.out_positions)
    il_path = os.path.join(out_dir, args.out_il)

    gif_dir_sampled = os.path.join(out_dir, "gifs")
    gif_dir_all = os.path.join(out_dir, "gifs_all")
    gif_dir_goal = os.path.join(out_dir, "gifs_goal")
    ensure_dir(gif_dir_sampled)
    ensure_dir(gif_dir_all)
    ensure_dir(gif_dir_goal)

    episodes_obs: List[np.ndarray] = []
    episodes_act: List[np.ndarray] = []
    episodes_rew: List[np.ndarray] = []
    episodes_done: List[np.ndarray] = []
    episodes_next_obs: List[np.ndarray] = []
    ep_returns_mail: List[float] = []
    ep_lengths_mail: List[int] = []

    traj_rows_csv: List[List] = []
    traj_ep_lengths: List[int] = []
    traj_delivery_flag: List[int] = []
    positions_rows: List[List] = []
    il_rows: List[List] = []

    ep_returns_all: List[float] = []
    ep_lengths_all: List[int] = []

    grid_shape = (5, 5)
    try:
        terrain = feat_env.mdp.terrain_mtx
        grid_shape = (len(terrain), len(terrain[0]))
    except Exception:
        pass

    for ep in range(args.episodes):
        obs_dict = feat_env.reset()

        delivery_happened = False
        delivery_t = None

        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        next_obs_buf = []

        gif_frames_full = []

        ep_ret = 0.0

        for t in range(args.episode_length):
            state = getattr(feat_env.env, "state", None)
            if state is not None:
                (x0, y0, d0, h0), (x1, y1, d1, h1) = _player_info_from_state(state)
            else:
                x0 = y0 = x1 = y1 = np.nan

            action_dict: Dict[str, int] = maddpg.select_action(obs_dict)
            actions_arr = np.array([int(action_dict[f"agent_{i}"]) for i in range(n_agents)], dtype=np.int32)

            next_obs_dict, reward_dict, done_dict, info_dict = feat_env.step(action_dict)

            sparse_now = float(reward_dict["agent_0"]) * 2.0
            if (sparse_now > 0) and (t >= int(args.min_delivery_step)):
                delivery_happened = True
                delivery_t = t

            if args.team_reward:
                team_r = float(sum(float(r) for r in reward_dict.values()))
                reward_arr = np.full((n_agents,), team_r, dtype=np.float32)
            else:
                reward_arr = np.array([float(reward_dict[f"agent_{i}"]) for i in range(n_agents)], dtype=np.float32)

            done_arr = np.array([bool(done_dict[f"agent_{i}"]) for i in range(n_agents)], dtype=np.bool_)
            obs_arr = np.stack([obs_dict[f"agent_{i}"] for i in range(n_agents)], axis=0).astype(np.float32)
            next_obs_arr = np.stack([next_obs_dict[f"agent_{i}"] for i in range(n_agents)], axis=0).astype(np.float32)

            obs_buf.append(obs_arr)
            act_buf.append(actions_arr)
            rew_buf.append(reward_arr)
            done_buf.append(done_arr)
            next_obs_buf.append(next_obs_arr)

            positions_rows.append([ep, t, "agent_0", x0, y0, int(action_dict["agent_0"]), np.nan, np.nan])
            positions_rows.append([ep, t, "agent_1", x1, y1, int(action_dict["agent_1"]), np.nan, np.nan])
            il_rows.append([
                ep, t, "agent_0",
                json.dumps(obs_dict["agent_0"].tolist()),
                int(action_dict["agent_0"]),
                json.dumps(next_obs_dict["agent_0"].tolist()),
                float(reward_dict["agent_0"]),
                int(bool(done_dict["agent_0"]))
            ])
            il_rows.append([
                ep, t, "agent_1",
                json.dumps(obs_dict["agent_1"].tolist()),
                int(action_dict["agent_1"]),
                json.dumps(next_obs_dict["agent_1"].tolist()),
                float(reward_dict["agent_1"]),
                int(bool(done_dict["agent_1"]))
            ])

            ep_ret += float(reward_arr.sum())

            state = getattr(feat_env.env, "state", None)
            if state is not None:
                (x0, y0, d0, h0), (x1, y1, d1, h1) = _player_info_from_state(state)

                event = "delivery" if (sparse_now > 0) else ""
                traj_rows_csv.append([ep, t, "player", 0, x0, y0, d0, h0, event])
                traj_rows_csv.append([ep, t, "player", 1, x1, y1, d1, h1, event])

                gif_frames_full.append({"t": t, "p0_xy": (x0, y0), "p1_xy": (x1, y1)})

            obs_dict = next_obs_dict

            if delivery_happened:
                break

            if bool(np.all(done_arr)):
                break

        obs_ep = np.stack(obs_buf, axis=0)
        act_ep = np.stack(act_buf, axis=0)
        rew_ep = np.stack(rew_buf, axis=0)
        done_ep = np.stack(done_buf, axis=0)
        next_obs_ep = np.stack(next_obs_buf, axis=0)
        T_ep = int(obs_ep.shape[0])

        traj_ep_lengths.append(T_ep)
        traj_delivery_flag.append(1 if delivery_happened else 0)
        ep_returns_all.append(ep_ret)
        ep_lengths_all.append(T_ep)

        keep_for_mail = True
        if args.only_delivery_episodes:
            keep_for_mail = bool(delivery_happened)

        if keep_for_mail:
            if bool(delivery_happened) and int(args.delivery_last_k) > 0 and delivery_t is not None:
                k = int(args.delivery_last_k)
                start = max(0, int(delivery_t) - k + 1)
                obs_ep = obs_ep[start:delivery_t+1]
                act_ep = act_ep[start:delivery_t+1]
                rew_ep = rew_ep[start:delivery_t+1]
                done_ep = done_ep[start:delivery_t+1]
                next_obs_ep = next_obs_ep[start:delivery_t+1]
                T_ep = int(obs_ep.shape[0])

            episodes_obs.append(obs_ep)
            episodes_act.append(act_ep)
            episodes_rew.append(rew_ep)
            episodes_done.append(done_ep)
            episodes_next_obs.append(next_obs_ep)
            ep_returns_mail.append(ep_ret)
            ep_lengths_mail.append(T_ep)

        if args.make_gif_all:
            gif_path_all = os.path.join(gif_dir_all, f"ep_{ep:04d}.gif")
            if try_make_gif_overcooked(gif_frames_full, gif_path_all, fps=args.gif_fps, grid_shape=grid_shape):
                print("[gif_all] saved:", gif_path_all)

        if args.make_gif_goal and delivery_happened:
            gif_path_goal = os.path.join(gif_dir_goal, f"ep_{ep:04d}_delivery.gif")
            if try_make_gif_overcooked(gif_frames_full, gif_path_goal, fps=args.gif_fps, grid_shape=grid_shape):
                print("[gif_goal] saved:", gif_path_goal)

        if args.make_gif and (ep % args.gif_every == 0):
            gif_path = os.path.join(gif_dir_sampled, f"ep_{ep:04d}.gif")
            if try_make_gif_overcooked(gif_frames_full, gif_path, fps=args.gif_fps, grid_shape=grid_shape):
                print("[gif] saved:", gif_path)

        if (ep + 1) % 10 == 0:
            mail_kept = len(ep_lengths_mail)
            print(f"[eval] ep={ep+1}/{args.episodes} len(all)={T_ep} "
                  f"return(sum over agents)={ep_ret:.3f} mail_kept={mail_kept}")

    feat_env.close()

    np.savez_compressed(
        mail_path,
        obs=np.array(episodes_obs, dtype=object),
        action=np.array(episodes_act, dtype=object),
        reward=np.array(episodes_rew, dtype=object),
        done=np.array(episodes_done, dtype=object),
        next_obs=np.array(episodes_next_obs, dtype=object),
        ep_return=np.array(ep_returns_mail, dtype=np.float32),
        ep_length=np.array(ep_lengths_mail, dtype=np.int32),
        obs_dim=np.int32(obs_dim),
        act_dim=np.int32(act_dim),
        n_agents=np.int32(n_agents),
    )

    np.savez_compressed(
        traj_path,
        traj_rows=np.array(traj_rows_csv, dtype=object),
        ep_length=np.array(traj_ep_lengths, dtype=np.int32),
        ep_delivery=np.array(traj_delivery_flag, dtype=np.int32),
        layout=args.layout,
    )

    df_pos = pd.DataFrame(
        positions_rows,
        columns=["episode", "t", "agent", "x", "y", "action", "ball_x", "ball_y"]
    )
    df_pos.to_csv(positions_path, index=False)

    df_il = pd.DataFrame(
        il_rows,
        columns=["episode", "t", "agent", "obs_json", "action", "next_obs_json", "reward", "done"]
    )
    df_il.to_csv(il_path, index=False)

    if args.export_csv:
        export_traj_csv_overcooked(traj_rows_csv, out_dir, filename="traj_entities.csv")
        export_mail_csv_head(
            episodes_obs, episodes_act, episodes_rew, episodes_done,
            out_dir,
            filename="mail_steps_head.csv",
            max_episodes=3,
            max_steps=int(args.mail_csv_head_steps),
        )

    print("[done] Saved MAIL dataset:", mail_path)
    print("[done] Saved TRAJ dataset:", traj_path)
    print("[done] Saved positions CSV:", positions_path)
    print("[done] Saved IL transitions CSV:", il_path)
    print("  model:", model_path)
    print("  episodes(all):", args.episodes)
    if len(ep_lengths_all) > 0:
        print("  avg_len(all):", float(np.mean(ep_lengths_all)))
        print("  avg_return(all,sum over agents):", float(np.mean(ep_returns_all)))
    if len(ep_lengths_mail) > 0:
        print("  kept_mail_eps:", len(ep_lengths_mail))
        print("  avg_len(mail_kept):", float(np.mean(ep_lengths_mail)))
        print("  avg_return(mail_kept,sum over agents):", float(np.mean(ep_returns_mail)))

    if args.save_meta:
        meta = {
            "layout": args.layout,
            "episode_length": args.episode_length,
            "shape_reward": bool(args.shape_reward),
            "team_reward_stored": bool(args.team_reward),
            "model_path": model_path,
            "episodes": args.episodes,
            "seed": args.seed,

            "only_delivery_episodes(mail_only)": bool(args.only_delivery_episodes),
            "delivery_last_k(mail_only)": int(args.delivery_last_k),
            "min_delivery_step": int(args.min_delivery_step),

            "make_gif_sampled": bool(args.make_gif),
            "gif_every": int(args.gif_every),
            "gif_fps": int(args.gif_fps),
            "make_gif_all": bool(args.make_gif_all),
            "make_gif_goal": bool(args.make_gif_goal),
            "grid_shape": list(grid_shape),
        }
        meta_path = os.path.join(out_dir, "eval_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("[done] Saved meta:", meta_path)


if __name__ == "__main__":
    main()
