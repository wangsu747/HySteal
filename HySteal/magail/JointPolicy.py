
import numpy as np
import torch
import torch.nn as nn

from hysteal.Memory import Memory
from models.mlp_actor import Actor


class JointPolicy(nn.Module):
    def __init__(self, initial_state, config=None):
        super(JointPolicy, self).__init__()
        self.config = config
        self.trajectory_length = int(config["trajectory_length"])

        self.agent_policy = nn.ModuleList([
            Actor(
                num_states=self.config["agent"]["num_states"],
                num_actions=self.config["agent"]["num_actions"],
                num_discrete_actions=self.config["agent"]["num_discrete_actions"],
                discrete_actions_sections=self.config["agent"]["discrete_actions_sections"],
                action_log_std=self.config["agent"]["action_log_std"],
                use_multivariate_distribution=self.config["agent"]["use_multivariate_distribution"],
                num_hiddens=self.config["agent"]["num_hiddens"],
                drop_rate=self.config["agent"]["drop_rate"],
                activation=self.config["agent"]["activation"],
                num_states_2=self.config["agent"]["num_states_2"],
            )
            for _ in range(self.config["agent"]["num_agent"])
        ])

        self.env_policy = nn.ModuleList([
            Actor(
                num_states=self.config["env"]["num_states"],
                num_actions=self.config["env"]["num_actions"],
                num_discrete_actions=self.config["env"]["num_discrete_actions"],
                discrete_actions_sections=self.config["env"]["discrete_actions_sections"],
                action_log_std=self.config["env"]["action_log_std"],
                use_multivariate_distribution=self.config["env"]["use_multivariate_distribution"],
                num_hiddens=self.config["env"]["num_hiddens"],
                drop_rate=self.config["env"]["drop_rate"],
                activation=self.config["env"]["activation"],
                num_states_2=self.config["env"]["num_states_2"],
            )
            for _ in range(self.config["env"]["num_agent"])
        ])

        self.initial_agent_state = initial_state

    def collect_samples(self, batch_size, num_agents, make_env_fn):

        memory = Memory()
        T = int(self.trajectory_length)

        parallel_size = (int(batch_size) + T - 1) // T
        dev = next(self.parameters()).device
        agent_state_device = dev
        envs = [make_env_fn() for _ in range(parallel_size)]

        base_seed = int(self.config.get("seed", 0) or 0)
        obs_list = []
        agents_order = None

        for i, e in enumerate(envs):
            obs, infos = e.reset(seed=base_seed + i)
            obs_list.append(obs)
            if agents_order is None:
                cfg_order = self.config.get("agent_order", None)
                if cfg_order is not None:
                    agents_order = list(cfg_order)

        cfg_order = self.config.get("agent_order", None)
        if cfg_order is not None:
            agents_order = list(cfg_order)

        assert len(agents_order) == num_agents, f"agents_order size mismatch: {len(agents_order)} vs {num_agents}"

        obs_dim = int(self.config["agent"]["num_states"])

        def pack_agent_state(obs_dict_list):
            obs_dim = self.config["agent"]["num_states"]
            X = np.zeros((parallel_size, num_agents, obs_dim), dtype=np.float32)

            for i in range(parallel_size):
                obs_i = obs_dict_list[i]

                a0 = np.asarray(obs_i["agent_0"], dtype=np.float32).reshape(-1)
                a0_vx, a0_vy = (a0[0], a0[1])

                for j, name in enumerate(agents_order):
                    vec = np.asarray(obs_i[name], dtype=np.float32).reshape(-1)

                    if vec.shape[0] >= obs_dim:
                        X[i, j, :] = vec[:obs_dim]
                    else:
                        L = min(vec.shape[0], obs_dim)
                        X[i, j, :L] = vec[:L]

                    if name == "agent_0" and obs_dim >= 16:
                        X[i, j, 14] = X[i, j, 0]
                        X[i, j, 15] = X[i, j, 1]

            X = torch.from_numpy(X).to(dtype=torch.float32, device=dev)
            return X.permute(1, 0, 2).contiguous()

        def decode_action(a_vec: torch.Tensor) -> int:
            if a_vec.dim() == 0:
                return int(a_vec.item())

            if a_vec.dim() == 2 and a_vec.shape[0] == 1:
                a_vec = a_vec.squeeze(0)

            if a_vec.numel() == 1:
                return int(a_vec.item())

            if a_vec.dim() != 1:
                raise RuntimeError(f"decode_action expects 1D vector, got shape={tuple(a_vec.shape)}")

            s = float(a_vec.sum().item())
            if abs(s - 1.0) < 1e-3 and float(a_vec.min().item()) >= -1e-6:
                return int(torch.argmax(a_vec, dim=-1).item())

            raise RuntimeError(
                f"decode_action got non-onehot vector (sum={s:.4f}, min={float(a_vec.min().item()):.4f}, max={float(a_vec.max().item()):.4f}). "
                f"Your Actor may be outputting logits/probs; need to SAMPLE action id consistently with log_prob."
            )

        self._debug_action_semantics = self.config.get("debug_action_semantics", True)
        self._debug_print_every = int(self.config.get("debug_print_every", 1))

        if not hasattr(self, "_dbg_dv_sum"):
            self._dbg_dv_sum = np.zeros((int(self.config["agent"]["num_actions"]), 2), dtype=np.float64)
            self._dbg_dv_cnt = np.zeros((int(self.config["agent"]["num_actions"]),), dtype=np.int64)

        def _dbg_update(obs_before: dict, obs_after: dict, act_dict: dict):
            if "agent_0" not in obs_before or "agent_0" not in obs_after:
                return
            a0_b = np.asarray(obs_before["agent_0"], dtype=np.float32).reshape(-1)
            a0_a = np.asarray(obs_after["agent_0"], dtype=np.float32).reshape(-1)
            if a0_b.shape[0] < 2 or a0_a.shape[0] < 2:
                return
            dv = a0_a[:2] - a0_b[:2]
            if "agent_0" not in act_dict:
                return
            aid = int(act_dict["agent_0"])
            if 0 <= aid < self._dbg_dv_sum.shape[0]:
                self._dbg_dv_sum[aid] += dv
                self._dbg_dv_cnt[aid] += 1

        def _dbg_print(prefix=""):
            if not self._debug_action_semantics:
                return
            lines = []
            for k in range(self._dbg_dv_sum.shape[0]):
                c = int(self._dbg_dv_cnt[k])
                if c == 0:
                    mean = (0.0, 0.0)
                else:
                    mean = (self._dbg_dv_sum[k][0] / c, self._dbg_dv_sum[k][1] / c)
                lines.append(f"a{k}: dv=({mean[0]:+.4f},{mean[1]:+.4f}) n={c}")
            print(prefix + " [DBG action dv] " + " | ".join(lines))

        agent_state = pack_agent_state(obs_list)

        for _t in range(1, T + 1):
            with torch.no_grad():
                global_state = agent_state.permute(1, 0, 2).reshape(parallel_size, -1)

                step_actions = []
                step_action_logps = []

                for j in range(num_agents):
                    a, a_logp = self.agent_policy[j].agent_get_action_log_prob(
                        agent_state[j],
                        global_state
                    )
                    step_actions.append(a)
                    step_action_logps.append(a_logp)

                actions_t = torch.stack(step_actions, dim=0)
                logps_t = torch.stack(step_action_logps, dim=0)

            next_obs_list = []
            mask_t = torch.ones((num_agents, parallel_size, 1), device=dev, dtype=torch.float32)

            for i, env in enumerate(envs):
                act_dict = {}
                for j, name in enumerate(agents_order):
                    act_dict[name] = decode_action(actions_t[j, i])

                obs_before = next_obs_list[i - 1] if False else obs_list[i]
                obs_before = obs_list[i]
                next_obs, rewards, terminations, truncations, infos = env.step(act_dict)
                _dbg_update(obs_before, next_obs, act_dict)

                done_i = any(terminations.values()) or any(truncations.values())
                if done_i:
                    mask_t[:, i, :] = 0.0
                    next_obs, infos = env.reset(seed=base_seed + 10000 + i)

                next_obs_list.append(next_obs)

            next_states_t = pack_agent_state(next_obs_list)

            memory.push(agent_state, actions_t, next_states_t, logps_t, mask_t)
            agent_state = next_states_t
        if self._debug_action_semantics:
            _dbg_print(prefix=f"[JointPolicy] T={T} parallel={parallel_size}")
        return memory.sample()

    def get_log_prob(self, states, actions, index_agent, global_states):
        return self.agent_policy[index_agent].get_log_prob(states, actions, global_states)

    def get_next_state(self, states, actions):
        raise NotImplementedError("Real-environment JointPolicy does not model next_state.")
