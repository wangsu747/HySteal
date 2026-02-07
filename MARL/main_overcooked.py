import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from MADDPG import MADDPG
from overcooked_wrapper import OvercookedParallelEnv


def get_running_reward(arr: np.ndarray, window=100):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        out[i] = np.mean(arr[max(0, i-window+1):i+1])
    return out


def make_result_dir(base="./results_overcooked", env_name="overcooked"):
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

    parser.add_argument("--layout", type=str, default="cramped_room",
                        choices=["cramped_room", "asymmetric_advantages", "coordination_ring"])
    parser.add_argument("--episode_num", type=int, default=30000)
    parser.add_argument("--episode_length", type=int, default=400)

    parser.add_argument("--learn_interval", type=int, default=50)
    parser.add_argument("--random_steps", type=int, default=10000)

    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_capacity", type=int, default=int(2e5))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)

    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=200000)

    parser.add_argument("--shape_reward", action="store_true")
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    env = OvercookedParallelEnv(
        layout_name=args.layout,
        episode_length=args.episode_length,
        shape_reward=bool(args.shape_reward),
        render=bool(args.render),
        quiet=True,
    )

    dim_info = {f"agent_{i}": [env.obs_dim, env.act_dim] for i in range(env.n_agents)}
    result_dir = make_result_dir("./results_overcooked", args.layout)

    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size,
                    args.actor_lr, args.critic_lr, result_dir)

    step = 0
    episode_sum_rewards = np.zeros(args.episode_num, dtype=np.float32)
    ep_deliveries = np.zeros(args.episode_num, dtype=np.int32)

    for ep in range(args.episode_num):
        obs = env.reset()
        agent_reward = {aid: 0.0 for aid in env.agents}
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

            maddpg.add(obs, action, reward, next_obs, done)

            for aid, r in reward.items():
                agent_reward[aid] += float(r)

            if step >= args.random_steps and step % args.learn_interval == 0:
                maddpg.learn(args.batch_size, args.gamma)
                maddpg.update_target(args.tau)

            obs = next_obs
            if all(done.values()):
                break

        s = sum(agent_reward.values())
        episode_sum_rewards[ep] = float(s)

        if last_stats is not None:
            ep_deliveries[ep] = int(last_stats.get("deliveries", 0))

        if (ep + 1) % 50 == 0:
            print(f"ep {ep+1}, eps={eps:.3f}, sum={s:.3f}, deliveries={ep_deliveries[ep]}")

    maddpg.save({"agent_0": episode_sum_rewards, "agent_1": episode_sum_rewards})

    x = np.arange(1, args.episode_num + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, episode_sum_rewards, label="sum_reward")
    plt.plot(x, get_running_reward(episode_sum_rewards, 200), label="sum_reward_ma200")
    plt.legend()
    plt.title(f"MADDPG Overcooked: {args.layout}")
    plt.savefig(os.path.join(result_dir, "train_curve.png"), dpi=150)

    plt.figure(figsize=(10, 5))
    plt.plot(x, get_running_reward(ep_deliveries.astype(np.float32), 200), label="deliveries_ma200")
    plt.legend()
    plt.title("Deliveries moving avg")
    plt.savefig(os.path.join(result_dir, "deliveries_curve.png"), dpi=150)

    env.close()
