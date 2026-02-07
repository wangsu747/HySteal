


import pandas as pd

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import gfootball.env as fe

from MADDPG import MADDPG
from grf_wrapper import GRFParallelEnv


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def done_from_done_dict(done_dict: Dict[str, bool]) -> bool:
    return bool(all(bool(v) for v in done_dict.values()))


def load_policy(env_name: str, folder: str, dim_info: Dict[str, List[int]]):
    model_dir = os.path.join("./results_grf", env_name, str(folder))
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model.pt: {model_path}")
    maddpg = MADDPG.load(dim_info, model_path)
    return maddpg, model_dir, model_path


def get_agent_xy_from_raw(
    raw_obs0: Dict[str, Any],
    agent_idx: int,
    n_left: int,
    n_right: int
) -> Tuple[float, float]:
    left_team = np.asarray(raw_obs0.get("left_team", []), dtype=np.float32)
    right_team = np.asarray(raw_obs0.get("right_team", []), dtype=np.float32)

    if agent_idx < n_left:
        pid = agent_idx
        if pid < len(left_team):
            return float(left_team[pid][0]), float(left_team[pid][1])
        return np.nan, np.nan

    pid = agent_idx - n_left
    if pid < n_right and pid < len(right_team):
        return float(right_team[pid][0]), float(right_team[pid][1])
    return np.nan, np.nan


def try_make_gif(
    frames,
    gif_path: str,
    fps: int = 10,
    n_left_ctrl: int = 2,
    n_right_ctrl: int = 0,
    show_all_teammates: bool = True,
):










    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if len(frames) == 0:
        return False

    XMIN, XMAX = -1.0, 1.0
    YMIN, YMAX = -0.42, 0.42

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.plot([XMIN, XMAX, XMAX, XMIN, XMIN], [YMIN, YMIN, YMAX, YMAX, YMIN], color="black", linewidth=1)
    ax.plot([0, 0], [YMIN, YMAX], color="gray", linewidth=1)
    ax.plot([XMAX, XMAX], [-0.15, 0.15], color="black", linewidth=3)

    ball_sc = ax.scatter([], [], s=35, c="black", marker="o", label="ball")
    left_ctrl_sc = ax.scatter([], [], s=70, c="red", marker="o", label="left(ctrl)")

    right_ctrl_sc = ax.scatter([], [], s=70, c="blue", marker="o", label="right(ctrl)")

    keeper_sc = ax.scatter([], [], s=120, c="blue", marker="s", label="keeper")

    left_other_sc = None
    right_other_sc = None
    if show_all_teammates:
        left_other_sc = ax.scatter([], [], s=25, c="lightgray", marker="o", alpha=0.5, label="left(other)")
        right_other_sc = ax.scatter([], [], s=25, c="lightskyblue", marker="o", alpha=0.5, label="right(other)")

    ball_line, = ax.plot([], [], color="black", linewidth=1, alpha=0.6)
    left_line, = ax.plot([], [], color="red", linewidth=1, alpha=0.6)
    right_line, = ax.plot([], [], color="blue", linewidth=1, alpha=0.6)

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def _empty_offsets():
        return np.zeros((0, 2), dtype=np.float32)

    def init():
        ball_sc.set_offsets(_empty_offsets())
        left_ctrl_sc.set_offsets(_empty_offsets())
        right_ctrl_sc.set_offsets(_empty_offsets())
        keeper_sc.set_offsets(_empty_offsets())

        if left_other_sc is not None:
            left_other_sc.set_offsets(_empty_offsets())
        if right_other_sc is not None:
            right_other_sc.set_offsets(_empty_offsets())

        ball_line.set_data([], [])
        left_line.set_data([], [])
        right_line.set_data([], [])
        title.set_text("")

        artists = [ball_sc, left_ctrl_sc, right_ctrl_sc, keeper_sc,
                   ball_line, left_line, right_line, title]
        if left_other_sc is not None:
            artists.append(left_other_sc)
        if right_other_sc is not None:
            artists.append(right_other_sc)
        return tuple(artists)

    def update(i):
        fr = frames[i]
        bx, by = fr["ball_xy"]
        left_xy_all = fr["left_xy"]
        right_xy_all = fr["right_xy"]

        left_xy = left_xy_all[:max(0, int(n_left_ctrl))]
        right_xy = right_xy_all[:max(0, int(n_right_ctrl))] if int(n_right_ctrl) > 0 else []

        keeper_xy = [right_xy_all[0]] if (isinstance(right_xy_all, list) and len(right_xy_all) > 0) else []

        ball_sc.set_offsets(np.array([[bx, by]], dtype=np.float32))
        left_ctrl_sc.set_offsets(np.array(left_xy, dtype=np.float32) if len(left_xy) else _empty_offsets())
        right_ctrl_sc.set_offsets(np.array(right_xy, dtype=np.float32) if len(right_xy) else _empty_offsets())
        keeper_sc.set_offsets(np.array(keeper_xy, dtype=np.float32) if len(keeper_xy) else _empty_offsets())

        if left_other_sc is not None:
            others_left = left_xy_all[max(0, int(n_left_ctrl)):]
            left_other_sc.set_offsets(np.array(others_left, dtype=np.float32) if len(others_left) else _empty_offsets())
        if right_other_sc is not None:
            others_right = right_xy_all[1:] if len(right_xy_all) > 1 else []
            right_other_sc.set_offsets(np.array(others_right, dtype=np.float32) if len(others_right) else _empty_offsets())

        ball_hist = np.array([f["ball_xy"] for f in frames[:i+1]], dtype=np.float32)
        ball_line.set_data(ball_hist[:, 0], ball_hist[:, 1])

        if int(n_left_ctrl) >= 1 and len(frames[0]["left_xy"]) >= 1:
            left_hist = np.array([frames[k]["left_xy"][0] for k in range(i+1)], dtype=np.float32)
            left_line.set_data(left_hist[:, 0], left_hist[:, 1])
        else:
            left_line.set_data([], [])

        if int(n_right_ctrl) >= 1 and len(frames[0]["right_xy"]) >= 1:
            right_hist = np.array([frames[k]["right_xy"][0] for k in range(i+1)], dtype=np.float32)
            right_line.set_data(right_hist[:, 0], right_hist[:, 1])
        else:
            right_line.set_data([], [])

        title.set_text(f"t={int(fr.get('t', i))}")
        artists = [ball_sc, left_ctrl_sc, right_ctrl_sc, keeper_sc,
                   ball_line, left_line, right_line, title]
        if left_other_sc is not None:
            artists.append(left_other_sc)
        if right_other_sc is not None:
            artists.append(right_other_sc)
        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True)

    ensure_dir(os.path.dirname(gif_path))
    anim.save(gif_path, writer=PillowWriter(fps=int(fps)))
    plt.close(fig)
    return True


def export_traj_csv(traj_ball, traj_left, traj_right, ep_lengths, out_dir: str, filename: str = "traj_entities.csv"):





    rows = []
    for ep in range(len(traj_ball)):
        T = int(ep_lengths[ep])
        ball_ep = traj_ball[ep]
        left_ep = traj_left[ep]
        right_ep = traj_right[ep]

        for t in range(T):
            bx, by = float(ball_ep[t][0]), float(ball_ep[t][1])
            rows.append([ep, t, "ball", -1, bx, by])

            for pid in range(left_ep[t].shape[0]):
                x, y = float(left_ep[t][pid][0]), float(left_ep[t][pid][1])
                rows.append([ep, t, "left", pid, x, y])

            for pid in range(right_ep[t].shape[0]):
                x, y = float(right_ep[t][pid][0]), float(right_ep[t][pid][1])
                rows.append([ep, t, "right", pid, x, y])

    df = pd.DataFrame(rows, columns=["episode", "t", "entity", "id", "x", "y"])
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
        obs_dim = obs_ep.shape[2]

        for t in range(T):
            for a in range(N):
                obs_head = obs_ep[t, a, :20].tolist()
                rows.append([ep, t, a, *obs_head, int(act_ep[t, a]), float(rew_ep[t, a]), bool(done_ep[t, a])])

    cols = ["episode", "t", "agent"] + [f"obs_{i}" for i in range(20)] + ["action", "reward", "done"]
    df = pd.DataFrame(rows, columns=cols)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    print("[csv] saved mail head:", out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="academy_empty_goal_close")
    parser.add_argument("--n_left", type=int, default=2)
    parser.add_argument("--n_right", type=int, default=0)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--rewards", type=str, default="scoring,checkpoints")

    parser.add_argument("--folder", type=str, required=True)

    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_positions", type=str, default="eval_positions_football.csv",
                        help="eval.py-compatible positions CSV")
    parser.add_argument("--out_il", type=str, default="il_transitions_football.csv",
                        help="eval.py-compatible IL transitions CSV")

    parser.add_argument("--out_mail", type=str, default="mail_dataset.npz")
    parser.add_argument("--out_traj", type=str, default="traj_dataset.npz")
    parser.add_argument("--save_meta", action="store_true")

    parser.add_argument("--use_shaping", action="store_true")
    parser.add_argument("--team_reward", action="store_true")

    parser.add_argument("--make_gif", action="store_true",
                        help="generate GIFs from raw trajectories (no rendering needed)")
    parser.add_argument("--gif_every", type=int, default=50,
                        help="make one gif every K episodes (e.g., 50 -> ep 0,50,100,150...)")
    parser.add_argument("--gif_fps", type=int, default=10)

    parser.add_argument("--export_csv", action="store_true",
                        help="also export CSV files alongside npz (traj_entities.csv, mail_steps_head.csv)")
    parser.add_argument("--mail_csv_head_steps", type=int, default=50,
                        help="when export_csv: export only first K steps per episode for mail_steps_head.csv")

    parser.add_argument("--only_goal_episodes", action="store_true",
                        help="Only keep episodes where left team scores for IL/MAIL/traj saving")
    parser.add_argument("--goal_last_k", type=int, default=0,
                        help="If >0, keep only last K steps before goal (including goal step) in IL data")
    parser.add_argument("--min_goal_step", type=int, default=0,
                        help="Ignore goals that happen too early (e.g., 0~2) to avoid trivial clips")

    args = parser.parse_args()
    np.random.seed(args.seed)

    feat_env = GRFParallelEnv(
        env_name=args.env_name,
        n_left=args.n_left,
        n_right=args.n_right,
        representation="simple115v2",
        rewards=args.rewards,
        render=False,
        episode_length=args.episode_length,
        shaping=bool(args.use_shaping),
    )

    n_agents = feat_env.n_agents
    obs_dim = feat_env.obs_dim
    act_dim = feat_env.act_dim
    dim_info = {f"agent_{i}": [obs_dim, act_dim] for i in range(n_agents)}

    maddpg, model_dir, model_path = load_policy(args.env_name, args.folder, dim_info)

    out_dir = os.path.join(model_dir, "eval")
    ensure_dir(out_dir)
    mail_path = os.path.join(out_dir, args.out_mail)
    traj_path = os.path.join(out_dir, args.out_traj)
    positions_path = os.path.join(out_dir, args.out_positions)
    il_path = os.path.join(out_dir, args.out_il)
    gif_dir = os.path.join(out_dir, "gifs")
    ensure_dir(gif_dir)

    raw_env = fe.create_environment(
        env_name=args.env_name,
        representation="raw",
        number_of_left_players_agent_controls=args.n_left,
        number_of_right_players_agent_controls=args.n_right,
        rewards=args.rewards,
        render=False,
    )

    episodes_obs: List[np.ndarray] = []
    episodes_act: List[np.ndarray] = []
    episodes_rew: List[np.ndarray] = []
    episodes_done: List[np.ndarray] = []
    episodes_next_obs: List[np.ndarray] = []
    ep_returns: List[float] = []
    ep_lengths: List[int] = []

    traj_ball: List[np.ndarray] = []
    traj_left: List[np.ndarray] = []
    traj_right: List[np.ndarray] = []
    positions_rows: List[List[Any]] = []
    il_rows: List[List[Any]] = []

    for ep in range(args.episodes):
        obs_dict = feat_env.reset()
        raw_obs = raw_env.reset()

        last_score_left, last_score_right = 0, 0
        goal_happened = False
        goal_t = None

        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        next_obs_buf = []

        ball_buf = []
        left_buf = []
        right_buf = []
        gif_frames = []

        ep_ret = 0.0

        for t in range(args.episode_length):
            raw0 = raw_obs[0] if isinstance(raw_obs, (list, tuple)) else raw_obs
            ball_cur = np.asarray(raw0.get("ball", [np.nan, np.nan]), dtype=np.float32)
            ball_x, ball_y = float(ball_cur[0]), float(ball_cur[1])

            action_dict: Dict[str, int] = maddpg.select_action(obs_dict)
            actions_arr = np.array([int(action_dict[f"agent_{i}"]) for i in range(n_agents)], dtype=np.int32)

            next_raw_obs, _, raw_done, raw_info = raw_env.step(actions_arr)
            next_obs_dict, reward_dict, done_dict, info_dict = feat_env.step(action_dict)
            info0 = raw_info[0] if isinstance(raw_info, (list, tuple)) else raw_info
            score = info0.get("score", None) if isinstance(info0, dict) else None

            if isinstance(score, (list, tuple)) and len(score) == 2:
                sl, sr = int(score[0]), int(score[1])
            else:
                sl, sr = last_score_left, last_score_right

            left_scored_now = (sl > last_score_left)
            if left_scored_now and (t >= int(args.min_goal_step)):
                goal_happened = True
                goal_t = t

            last_score_left, last_score_right = sl, sr

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

            for i in range(n_agents):
                aid = f"agent_{i}"
                ax = int(action_dict[aid])
                x, y = get_agent_xy_from_raw(raw0, i, n_left=args.n_left, n_right=args.n_right)
                positions_rows.append([ep, t, aid, x, y, ax, ball_x, ball_y])
                il_rows.append([
                    ep,
                    t,
                    aid,
                    json.dumps(obs_dict[aid].tolist()),
                    ax,
                    json.dumps(next_obs_dict[aid].tolist()),
                    float(reward_dict[aid]),
                    int(bool(done_dict[aid]))
                ])

            ep_ret += float(reward_arr.sum())

            ball = next_raw_obs[0]["ball"]
            left_team = next_raw_obs[0]["left_team"]
            right_team = next_raw_obs[0]["right_team"]

            bx, by = float(ball[0]), float(ball[1])
            ball_buf.append([bx, by])
            left_buf.append(np.array(left_team, dtype=np.float32))
            right_buf.append(np.array(right_team, dtype=np.float32))

            if args.make_gif and (ep % args.gif_every == 0):
                gif_frames.append({
                    "t": t,
                    "ball_xy": (bx, by),
                    "left_xy": [(float(left_team[i][0]), float(left_team[i][1])) for i in range(left_team.shape[0])],
                    "right_xy": [(float(right_team[i][0]), float(right_team[i][1])) for i in range(right_team.shape[0])],
                })

            obs_dict = next_obs_dict
            raw_obs = next_raw_obs

            if goal_happened:
                break

            if bool(raw_done) or bool(np.all(done_arr)):
                break

        obs_ep = np.stack(obs_buf, axis=0)
        act_ep = np.stack(act_buf, axis=0)
        rew_ep = np.stack(rew_buf, axis=0)
        done_ep = np.stack(done_buf, axis=0)
        next_obs_ep = np.stack(next_obs_buf, axis=0)

        keep_for_il = True
        if args.only_goal_episodes:
            keep_for_il = bool(goal_happened)

        if keep_for_il:
            episodes_obs.append(obs_ep)
            episodes_act.append(act_ep)
            episodes_rew.append(rew_ep)
            episodes_done.append(done_ep)
            episodes_next_obs.append(next_obs_ep)

            ep_returns.append(ep_ret)
            ep_lengths.append(int(obs_ep.shape[0]))

            traj_ball.append(np.array(ball_buf, dtype=np.float32))
            traj_left.append(np.stack(left_buf, axis=0))
            traj_right.append(np.stack(right_buf, axis=0))

        if args.make_gif and (ep % args.gif_every == 0):
            gif_path = os.path.join(gif_dir, f"ep_{ep:04d}.gif")
            ok = try_make_gif(gif_frames, gif_path, fps=args.gif_fps,
                              n_left_ctrl=args.n_left, n_right_ctrl=args.n_right,
                              show_all_teammates=True)

            if ok:
                print(f"[gif] saved: {gif_path}")

        if (ep + 1) % 10 == 0:
            print(f"[eval] ep={ep+1}/{args.episodes} len={ep_lengths[-1]} return(sum over agents)={ep_ret:.3f}")

    feat_env.close()
    raw_env.close()

    np.savez_compressed(
        mail_path,
        obs=np.array(episodes_obs, dtype=object),
        action=np.array(episodes_act, dtype=object),
        reward=np.array(episodes_rew, dtype=object),
        done=np.array(episodes_done, dtype=object),
        next_obs=np.array(episodes_next_obs, dtype=object),
        ep_return=np.array(ep_returns, dtype=np.float32),
        ep_length=np.array(ep_lengths, dtype=np.int32),
        obs_dim=np.int32(obs_dim),
        act_dim=np.int32(act_dim),
        n_agents=np.int32(n_agents),
    )

    np.savez_compressed(
        traj_path,
        ball=np.array(traj_ball, dtype=object),
        left=np.array(traj_left, dtype=object),
        right=np.array(traj_right, dtype=object),
        ep_length=np.array(ep_lengths, dtype=np.int32),
        env_name=args.env_name,
        n_left=np.int32(args.n_left),
        n_right=np.int32(args.n_right),
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
        export_traj_csv(
            traj_ball,
            traj_left,
            traj_right,
            ep_lengths,
            out_dir,
            filename="traj_entities.csv"
        )

        export_mail_csv_head(
            episodes_obs,
            episodes_act,
            episodes_rew,
            episodes_done,
            out_dir,
            filename="mail_steps_head.csv",
            max_episodes=3,
            max_steps=int(args.mail_csv_head_steps)
        )

    print("[done] Saved MAIL dataset:", mail_path)
    print("[done] Saved trajectory dataset:", traj_path)
    print("[done] Saved positions CSV:", positions_path)
    print("[done] Saved IL transitions CSV:", il_path)
    print("  model:", model_path)
    print("  episodes:", args.episodes)
    print("  avg_len:", float(np.mean(ep_lengths)))
    print("  avg_return(sum over agents):", float(np.mean(ep_returns)))

    if args.save_meta:
        meta = {
            "env_name": args.env_name,
            "n_left": args.n_left,
            "n_right": args.n_right,
            "episode_length": args.episode_length,
            "rewards": args.rewards,
            "use_shaping": bool(args.use_shaping),
            "team_reward_stored": bool(args.team_reward),
            "model_path": model_path,
            "episodes": args.episodes,
            "seed": args.seed,
            "make_gif": bool(args.make_gif),
            "gif_every": int(args.gif_every),
            "gif_fps": int(args.gif_fps),
        }
        meta_path = os.path.join(out_dir, "eval_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("[done] Saved meta:", meta_path)


if __name__ == "__main__":
    main()
