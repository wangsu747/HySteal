import logging
import os
import pickle
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == handler.baseFilename
               for h in logger.handlers):
        logger.addHandler(handler)

    return logger


def _get_optim(agent: Agent, name_candidates):

    for n in name_candidates:
        if hasattr(agent, n):
            return getattr(agent, n)
    return None


class MADDPG:


    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir):
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())

        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu')

        self.dim_info = dim_info
        self.batch_size = batch_size
        self.res_dir = res_dir
        os.makedirs(res_dir, exist_ok=True)
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

        self.global_step = 0

    def add(self, obs, action, reward, next_obs, done, triggered=None, step_in_episode=None):
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            if isinstance(a, int):
                a = np.eye(self.dim_info[agent_id][1])[a]
            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            trg = False if triggered is None else bool(triggered.get(agent_id, False))
            stp = 0 if step_in_episode is None else int(step_in_episode.get(agent_id, 0))
            self.buffers[agent_id].add(o, a, r, next_o, d, triggered=trg, step_in_episode=stp)

    def sample(self, batch_size):
        total_num = len(next(iter(self.buffers.values())))
        if total_num < batch_size:
            return None
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        obs, act, reward, next_obs, done, next_act, triggered, step_in_episode = {}, {}, {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d, trg, stp = buffer.sample(indices)
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            triggered[agent_id] = trg
            step_in_episode[agent_id] = stp
            next_act[agent_id] = self.agents[agent_id].target_action(n_o).detach()

        return obs, act, reward, next_obs, done, next_act, triggered, step_in_episode

    def select_action(self, obs):
        actions = {}
        for agent_id, o in obs.items():
            o_t = torch.from_numpy(o).unsqueeze(0).float()
            a = self.agents[agent_id].action(o_t)
            actions[agent_id] = a.squeeze(0).argmax().item()
        return actions

    def learn(
        self,
        batch_size,
        gamma,
        wm_cfg: Optional[Dict[str, Any]] = None,
        param_wm_cfg: Optional[Dict[str, Any]] = None
    ):
        sample = self.sample(batch_size)
        if sample is None:
            return

        obs, act, reward, next_obs, done, next_act, triggered, step_in_episode = sample

        for agent_id, agent in self.agents.items():
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))
            with torch.no_grad():
                next_target_critic_value = agent.target_critic_value(
                    list(next_obs.values()), list(next_act.values())
                )
                target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value, reduction='mean')
            agent.update_critic(critic_loss)

            action_i, logits = agent.action(obs[agent_id], model_out=True)

            joint_act = []
            for aid in self.agents.keys():
                if aid == agent_id:
                    joint_act.append(action_i)
                else:
                    joint_act.append(act[aid].detach())

            actor_loss = -agent.critic_value(list(obs.values()), joint_act).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            total_actor_loss = actor_loss + 1e-3 * actor_loss_pse

            if wm_cfg and wm_cfg.get("enabled", False):
                trg_mask = triggered[agent_id] > 0.5
                if int(trg_mask.sum().item()) > 0:
                    act_n = int(self.dim_info[agent_id][1])
                    if wm_cfg.get("target_mode", "fixed") == "seq":
                        base = int(wm_cfg.get("seq_base", {}).get(agent_id, 0))
                        tgt = (base + step_in_episode[agent_id]) % act_n
                    else:
                        fixed = int(wm_cfg.get("target_actions", {}).get(agent_id, 0))
                        tgt = torch.full_like(step_in_episode[agent_id], fixed, dtype=torch.long)
                    wm_loss = F.cross_entropy(logits[trg_mask], tgt[trg_mask])
                    total_actor_loss = total_actor_loss + float(wm_cfg.get("lambda_wm", 1.0)) * wm_loss

            if param_wm_cfg and param_wm_cfg.get("enabled", False):
                per_agent = param_wm_cfg.get("per_agent", {})
                if agent_id in per_agent:
                    cfg = per_agent[agent_id]
                    R = cfg["R"].to(logits.device)
                    b = cfg["b"].to(logits.device)
                    w = agent.actor.net[-1].weight.view(-1)
                    z = torch.matmul(R, w)
                    wm_param_loss = F.binary_cross_entropy_with_logits(z, b)
                    total_actor_loss = total_actor_loss + float(param_wm_cfg.get("lambda_wm", 1.0)) * wm_param_loss

            agent.update_actor(total_actor_loss)

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save_checkpoint(self,
                        episode_rewards=None,
                        filename: str = "checkpoint.pt",
                        save_buffers: bool = False):








        ckpt: Dict[str, Any] = {
            "dim_info": self.dim_info,
            "global_step": int(self.global_step),
            "agents": {},
        }

        for aid, agent in self.agents.items():
            actor_opt = _get_optim(agent, ["actor_optimizer", "actor_optim", "actor_opt", "optim_actor"])
            critic_opt = _get_optim(agent, ["critic_optimizer", "critic_optim", "critic_opt", "optim_critic"])

            ckpt["agents"][aid] = {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "target_actor": agent.target_actor.state_dict(),
                "target_critic": agent.target_critic.state_dict(),
                "actor_opt": actor_opt.state_dict() if actor_opt is not None else None,
                "critic_opt": critic_opt.state_dict() if critic_opt is not None else None,
            }

        if save_buffers:
            ckpt["buffers"] = self.buffers

        out_path = os.path.join(self.res_dir, filename)
        torch.save(ckpt, out_path)

        if episode_rewards is not None:
            with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:
                pickle.dump({'rewards': episode_rewards}, f)

        return out_path

    def load_checkpoint_into_self(self, ckpt_path: str, load_buffers: bool = False):




        data = torch.load(ckpt_path, map_location="cpu")
        self.global_step = int(data.get("global_step", 0))

        agents_blob = data.get("agents", {})
        for aid, blob in agents_blob.items():
            if aid not in self.agents:
                continue
            agent = self.agents[aid]
            agent.actor.load_state_dict(blob["actor"])
            agent.critic.load_state_dict(blob["critic"])
            agent.target_actor.load_state_dict(blob["target_actor"])
            agent.target_critic.load_state_dict(blob["target_critic"])

            actor_opt = _get_optim(agent, ["actor_optimizer", "actor_optim", "actor_opt", "optim_actor"])
            critic_opt = _get_optim(agent, ["critic_optimizer", "critic_optim", "critic_opt", "optim_critic"])

            if actor_opt is not None and blob.get("actor_opt", None) is not None:
                actor_opt.load_state_dict(blob["actor_opt"])
            if critic_opt is not None and blob.get("critic_opt", None) is not None:
                critic_opt.load_state_dict(blob["critic_opt"])

        if load_buffers and ("buffers" in data):
            self.buffers = data["buffers"]

    def save(self, reward):
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file, map_location="cpu")
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
