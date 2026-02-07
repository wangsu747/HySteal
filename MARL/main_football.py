import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from MADDPG import MADDPG
from grf_wrapper import GRFParallelEnv


def get_running_reward(arr: np.ndarray, window=100):
    running_reward = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        running_reward[i] = np.mean(arr[max(0, i - window + 1): i + 1])
    return running_reward


def make_result_dir(base="./results_grf", env_name="grf"):
    env_dir = os.path.join(base, env_name)
    os.makedirs(env_dir, exist_ok=True)
    total_files = len([f for f in os.listdir(env_dir) if os.path.isdir(os.path.join(env_dir, f))])
    result_dir = os.path.join(env_dir, f"{total_files + 1}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def epsilon_by_step(step, eps_start, eps_end, eps_decay):
    return float(eps_end + (eps_start - eps_end) * np.exp(-step / eps_decay))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--n_left", type=int, default=3)
    parser.add_argument("--n_right", type=int, default=0)
    parser.add_argument("--representation", type=str, default="simple115v2",
                        choices=["simple115v2", "simple115"])
    parser.add_argument("--rewards", type=str, default="scoring,checkpoints")
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--episode_num", type=int, default=40000)
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--learn_interval", type=int, default=50)
    parser.add_argument("--random_steps", type=int, default=20000)

    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_capacity", type=int, default=int(2e5))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)

    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=200000)

    parser.add_argument("--team_reward", action="store_true")

    parser.add_argument("--no_shaping", action="store_true")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_folder", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="checkpoint.pt",
                        help="try load this checkpoint name first inside resume folder")
    parser.add_argument("--load_buffers", action="store_true",
                        help="also load replay buffers from checkpoint (only if checkpoint saved them)")

    parser.add_argument("--save_every", type=int, default=1000,
                        help="save full checkpoint every N episodes (0=disable)")
    parser.add_argument("--save_buffers", action="store_true",
                        help="save replay buffers into checkpoint (may be large / may fail if Buffer not picklable)")

    args = parser.parse_args()

    env = GRFParallelEnv(
        env_name=args.env_name,
        n_left=args.n_left,
        n_right=args.n_right,
        representation=args.representation,
        rewards=args.rewards,
        render=args.render,
        episode_length=args.episode_length,
        shaping=(not args.no_shaping),
    )

    dim_info = {f"agent_{i}": [env.obs_dim, env.act_dim] for i in range(env.n_agents)}

    result_dir = make_result_dir("./results_grf", args.env_name)

    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size,
                    args.actor_lr, args.critic_lr, result_dir)

    step = 0
    if args.resume:
        resume_dir = os.path.join("./results_grf", args.env_name, str(args.resume_folder))
        ckpt_path = os.path.join(resume_dir, args.resume_ckpt)
        model_path = os.path.join(resume_dir, "model.pt")

        if os.path.exists(ckpt_path):
            print("Resuming FULL checkpoint from:", ckpt_path)
            maddpg.load_checkpoint_into_self(ckpt_path, load_buffers=bool(args.load_buffers))
            step = int(maddpg.global_step)
            if step < args.random_steps:
                step = int(args.random_steps)
                maddpg.global_step = step
        elif os.path.exists(model_path):
            print("Resuming ACTOR-only from:", model_path)
            loaded = MADDPG.load(dim_info, model_path)
            for aid in maddpg.agents.keys():
                maddpg.agents[aid].actor.load_state_dict(loaded.agents[aid].actor.state_dict())
                maddpg.agents[aid].target_actor.load_state_dict(maddpg.agents[aid].actor.state_dict())
            step = int(args.random_steps)
            maddpg.global_step = step
        else:
            raise FileNotFoundError(f"Cannot find resume checkpoint/model in {resume_dir}")

    episode_rewards = {agent_id: np.zeros(args.episode_num, dtype=np.float32) for agent_id in dim_info.keys()}
    episode_sum_rewards = np.zeros(args.episode_num, dtype=np.float32)

    ep_shots = np.zeros(args.episode_num, dtype=np.int32)
    ep_goals = np.zeros(args.episode_num, dtype=np.int32)
    ep_passes = np.zeros(args.episode_num, dtype=np.int32)
    ep_poss = np.zeros(args.episode_num, dtype=np.int32)
    ep_prog = np.zeros(args.episode_num, dtype=np.float32)

    for episode in range(args.episode_num):
        obs = env.reset()
        agent_reward = {agent_id: 0.0 for agent_id in env.agents}
        last_stats = None

        for t in range(args.episode_length):
            step += 1
            maddpg.global_step = step

            eps = epsilon_by_step(step, args.eps_start, args.eps_end, args.eps_decay)

            if step < args.random_steps:
                action = env.sample_action_dict()
            else:
                action = maddpg.select_action(obs)

                rand_action = env.sample_action_dict()
                for aid in env.agents:
                    if np.random.rand() < eps:
                        action[aid] = rand_action[aid]

            next_obs, reward, done, info = env.step(action)
            last_stats = info["agent_0"].get("maddpg_stats", None)

            if args.team_reward:
                team_r = float(sum(reward.values())) / float(len(reward))
                reward = {aid: team_r for aid in reward.keys()}

            maddpg.add(obs, action, reward, next_obs, done)

            for aid, r in reward.items():
                agent_reward[aid] += float(r)

            if step >= args.random_steps and step % args.learn_interval == 0:
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs
            if all(done.values()):
                break

        s = 0.0
        for aid, r in agent_reward.items():
            episode_rewards[aid][episode] = r
            s += r
        episode_sum_rewards[episode] = s

        if last_stats is not None:
            ep_shots[episode] = int(last_stats.get("shots", 0))
            ep_goals[episode] = int(last_stats.get("goals_for", 0))
            ep_passes[episode] = int(last_stats.get("passes", 0))
            ep_poss[episode] = int(last_stats.get("possession_steps", 0))
            ep_prog[episode] = float(last_stats.get("ball_progress", 0.0))

        if (episode + 1) % 50 == 0:
            msg = f"episode {episode+1}, eps={eps:.3f}, "
            for aid in env.agents:
                msg += f"{aid}: {agent_reward[aid]:.3f}; "
            msg += f"sum: {s:.3f} | shots={ep_shots[episode]} goals={ep_goals[episode]} passes={ep_passes[episode]} poss={ep_poss[episode]} prog={ep_prog[episode]:.3f}"
            print(msg)

        if args.save_every > 0 and (episode + 1) % args.save_every == 0:
            ckpt_file = maddpg.save_checkpoint(
                episode_rewards=None,
                filename="checkpoint.pt",
                save_buffers=bool(args.save_buffers),
            )
            print("[ckpt] saved:", ckpt_file)

    ckpt_file = maddpg.save_checkpoint(
        episode_rewards=episode_rewards,
        filename="checkpoint.pt",
        save_buffers=bool(args.save_buffers),
    )
    print("[ckpt] final saved:", ckpt_file)

    maddpg.save(episode_rewards)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, args.episode_num + 1)
    ax.plot(x, episode_sum_rewards, label="sum_reward")
    ax.plot(x, get_running_reward(episode_sum_rewards, 100), label="sum_reward_ma100")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title(f"MADDPG on GRF: {args.env_name} (n_agents={env.n_agents})")
    ax.legend()
    out = os.path.join(result_dir, "train_curve.png")
    plt.savefig(out, dpi=150)
    print("Saved plot:", out)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x, get_running_reward(ep_shots.astype(np.float32), 50), label="shots_ma50")
    ax2.plot(x, get_running_reward(ep_passes.astype(np.float32), 50), label="passes_ma50")
    ax2.plot(x, get_running_reward(ep_poss.astype(np.float32), 50), label="possession_steps_ma50")
    ax2.plot(x, get_running_reward(ep_prog, 50), label="ball_progress_ma50")
    ax2.plot(x, get_running_reward(ep_goals.astype(np.float32), 50), label="goals_ma50")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("metric")
    ax2.set_title("Makes-sense metrics (moving avg)")
    ax2.legend()
    out2 = os.path.join(result_dir, "metrics_curve.png")
    plt.savefig(out2, dpi=150)
    print("Saved metrics plot:", out2)

    env.close()
