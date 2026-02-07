


import math
import multiprocessing
import os
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hysteal.JointPolicy import JointPolicy
from hysteal.ppo_step import ppo_step

from data.ExpertCSVDataset import ExpertCSVDataset
from models.mlp_critic import Value
from models.transformer_discriminator import MILD2TransformerDiscriminator

from utils.torch_util import device, to_device
from pettingzoo.mpe import simple_tag_v3
from magail.dynamics_env import DynamicsEnv


def stat(x, name):
    x = x.detach()
    print(f"[STAT] {name}: mean={x.mean().item():+.4f} std={x.std().item():.4f} "
          f"min={x.min().item():+.4f} max={x.max().item():+.4f}")

def _as_int(x, name=""):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return int(s)
        except ValueError:
            return int(float(s))
    raise TypeError(f"{name} must be int/float(str-int), got {type(x)}: {x}")


def _as_float(x, name=""):
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Cannot convert {name}='{x}' to float")
    raise TypeError(f"{name} must be float/int/str, got {type(x)}: {x}")


def _get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def multi_trans_shape_func(batch_states: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:





    stacked = torch.stack(batch_states)
    trans = stacked.transpose(0, 1)
    n_agents = trans.shape[0]
    dim = trans.shape[-1]
    out = tuple(trans[i].reshape(-1, dim) for i in range(n_agents))
    return out


def to_onehot_from_ids(action_ids: torch.Tensor, action_n: int) -> torch.Tensor:




    if action_ids.dim() == 2 and action_ids.shape[1] == 1:
        action_ids = action_ids.squeeze(1)
    action_ids = action_ids.long()
    B = action_ids.shape[0]
    oh = torch.zeros(B, action_n, device=action_ids.device, dtype=torch.float32)
    valid = (action_ids >= 0) & (action_ids < action_n)
    if valid.any():
        oh[torch.arange(B, device=action_ids.device)[valid], action_ids[valid]] = 1.0
    return oh


def _ensure_onehot_action(a: torch.Tensor, action_n: int) -> torch.Tensor:







    if a.dim() == 2 and a.shape[1] == action_n:
        return a.float()
    if a.dim() == 2 and a.shape[1] == 1:
        return to_onehot_from_ids(a, action_n)
    if a.dim() == 1:
        return to_onehot_from_ids(a, action_n)
    return to_onehot_from_ids(torch.argmax(a, dim=-1), action_n)

def _to_float(x):
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)

def gae_from_rewards(
    rewards: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
    lam: float,
    T: int,
):




    assert rewards.dim() == 2 and rewards.shape[1] == 1
    assert values.shape == rewards.shape
    assert masks.shape == rewards.shape

    TP = rewards.shape[0]
    assert TP % T == 0, f"TP={TP} must be multiple of T={T}"
    P = TP // T

    r = rewards.view(T, P, 1)
    v = values.view(T, P, 1)
    m = masks.view(T, P, 1).float()

    adv = torch.zeros_like(r)
    last_gae = torch.zeros((P, 1), device=rewards.device)
    last_v = torch.zeros((P, 1), device=rewards.device)

    for t in reversed(range(T)):
        next_v = last_v if t == T - 1 else v[t + 1]
        delta = r[t] + gamma * next_v * m[t] - v[t]
        last_gae = delta + gamma * lam * last_gae * m[t]
        adv[t] = last_gae

    ret = v + adv
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return adv.view(TP, 1), ret.view(TP, 1)


class HySteal:









    def __init__(self, config: Dict[str, Any], log_dir: str, exp_name: str, num_agent: int, bc_init: str = None):
        self.config = config
        self.exp_name = exp_name
        self.num_agent = int(num_agent)
        self.bc_init = bc_init

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.exp_name}")

        seed = _as_int(self.config["general"].get("seed", 0), "general.seed")
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._load_expert_data()
        self._init_model()

    def _load_expert_data(self):
        g = self.config["general"]
        self.obs_dim_per_agent = _as_int(g["obs_dim_per_agent"], "general.obs_dim_per_agent")
        self.action_n = _as_int(g["action_n"], "general.action_n")
        self.agent_order = list(g["agent_order"])
        self.expert_batch_size = _as_int(g["expert_batch_size"], "general.expert_batch_size")
        self.expert_csv_path = g["expert_csv_path"]

        self.expert_dataset = ExpertCSVDataset(
            csv_path=self.expert_csv_path,
            agent_order=self.agent_order,
            obs_dim=self.obs_dim_per_agent,
            action_n=self.action_n,
            require_full=True,
        )

        nw = min(max(1, multiprocessing.cpu_count() // 2), 8)
        self.expert_data_loader = DataLoader(
            dataset=self.expert_dataset,
            batch_size=self.expert_batch_size,
            shuffle=True,
            num_workers=nw,
            drop_last=True,
            pin_memory=False,
        )
        self._expert_iter = iter(self.expert_data_loader)

    def _next_expert_batch(self):
        try:
            batch = next(self._expert_iter)
        except StopIteration:
            self._expert_iter = iter(self.expert_data_loader)
            batch = next(self._expert_iter)
        return batch

    def _build_make_env_fn(self):
        cfg_order = list(self.config.get("general", {}).get("agent_order", []))
        n_adv = sum(1 for a in cfg_order if str(a).startswith("adversary_"))
        n_good = sum(1 for a in cfg_order if str(a).startswith("agent_"))
        use_dynamics_env = bool(self.config.get("general", {}).get("use_dynamics_env", False))
        dynamics_ckpt = str(self.config.get("general", {}).get("dynamics_ckpt_path", "")).strip()

        def make_env():
            if use_dynamics_env:
                if not dynamics_ckpt:
                    raise ValueError("general.dynamics_ckpt_path is required when use_dynamics_env is True")
                return DynamicsEnv(
                    dynamics_ckpt,
                    episode_length=self.config["jointpolicy"]["trajectory_length"],
                    device="auto",
                )
            kwargs = dict(
                max_cycles=self.config["jointpolicy"]["trajectory_length"],
                continuous_actions=False,
            )
            if n_adv > 0 and n_good > 0:
                kwargs["num_adversaries"] = int(n_adv)
                kwargs["num_good"] = int(n_good)
            try:
                env = simple_tag_v3.parallel_env(**kwargs)
            except TypeError:
                kwargs.pop("num_adversaries", None)
                kwargs.pop("num_good", None)
                env = simple_tag_v3.parallel_env(**kwargs)
            return env
        return make_env

    def _init_model(self):
        self._policy_lr = _as_float(_get(self.config, "jointpolicy.learning_rate"), "jointpolicy.learning_rate")
        self._value_lr  = _as_float(_get(self.config, "value.learning_rate"), "value.learning_rate")
        self._disc_lr   = _as_float(_get(self.config, "discriminator.learning_rate"), "discriminator.learning_rate")
        self._l2_reg    = _as_float(_get(self.config, "value.l2_reg", 0.0), "value.l2_reg")

        self._clip_ratio = _as_float(_get(self.config, "ppo.clip_ratio"), "ppo.clip_ratio")
        self._gamma      = _as_float(_get(self.config, "gae.gamma"), "gae.gamma")
        self._tau        = _as_float(_get(self.config, "gae.tau"), "gae.tau")

        agent_state_dim = self.obs_dim_per_agent
        agent_action_dim = self.action_n
        global_state_dim = self.num_agent * agent_state_dim

        self.V = [
            Value(
                num_states=agent_state_dim,
                num_hiddens=self.config["value"]["num_hiddens"],
                drop_rate=self.config["value"]["drop_rate"],
                activation=self.config["value"]["activation"],
            )
            for _ in range(self.num_agent)
        ]

        self.V_g = Value(
            num_states=global_state_dim,
            num_hiddens=self.config["value"]["num_hiddens"],
            drop_rate=self.config["value"]["drop_rate"],
            activation=self.config["value"]["activation"],
        )

        if hasattr(self.expert_dataset, "states"):
            init_pool = self.expert_dataset.states
        elif hasattr(self.expert_dataset, "state"):
            init_pool = self.expert_dataset.state
        else:
            raise AttributeError("ExpertCSVDataset must have .states or .state tensor (global states)")

        init_pool = init_pool.to(device)
        self.config["jointpolicy"]["agent_order"] = self.agent_order
        self.P = JointPolicy(initial_state=init_pool, config=self.config["jointpolicy"])

        self.optimizer_policy = optim.Adam(self.P.agent_policy.parameters(), lr=self._policy_lr)
        self.optimizer_value  = [optim.Adam(v.parameters(), lr=self._value_lr) for v in self.V]

        td_cfg = self.config.get("transformer_discriminator", {})
        self.gp_lambda = float(td_cfg.get("gp_lambda", 10.0))
        d_model = int(td_cfg.get("d_model", 128))
        nhead = int(td_cfg.get("nhead", 4))
        enc_layers = int(td_cfg.get("enc_layers", 2))
        dec_layers = int(td_cfg.get("dec_layers", 2))
        ff = int(td_cfg.get("dim_feedforward", 256))
        drop = float(td_cfg.get("dropout", 0.1))
        use_agent_id = bool(td_cfg.get("use_agent_id_embedding", True))

        self.TD = MILD2TransformerDiscriminator(
            obs_dim=agent_state_dim,
            act_dim=agent_action_dim,
            n_agents=self.num_agent,
            d_model=d_model,
            nhead=nhead,
            num_enc_layers=enc_layers,
            num_dec_layers=dec_layers,
            dim_feedforward=ff,
            dropout=drop,
            use_agent_id_embedding=use_agent_id,
        )

        self.optimizer_td = optim.Adam(self.TD.parameters(), lr=self._disc_lr)
        self.scheduler_td = optim.lr_scheduler.StepLR(self.optimizer_td, step_size=2000, gamma=0.95)

        to_device(*self.V, self.V_g, self.P, self.TD)
        self.make_env_fn = self._build_make_env_fn()

    def _disc_bce_loss(self, exp_obs_tok, exp_act_tok, gen_obs_tok, gen_act_tok):
        Be = exp_obs_tok.size(0)
        Bg = gen_obs_tok.size(0)
        B = min(Be, Bg)

        exp_obs_tok = exp_obs_tok[:B]
        exp_act_tok = exp_act_tok[:B]
        gen_obs_tok = gen_obs_tok[:B]
        gen_act_tok = gen_act_tok[:B]

        logits_exp = self.TD(exp_obs_tok, exp_act_tok)
        logits_gen = self.TD(gen_obs_tok, gen_act_tok)

        label_smooth = float(self.config.get("discriminator", {}).get("label_smooth", 0.0))
        loss_td, loss_exp, loss_gen = MILD2TransformerDiscriminator.bce_loss(
            logits_exp, logits_gen, label_smooth=label_smooth
        )

        with torch.no_grad():
            e_mean = torch.sigmoid(logits_exp).mean().item()
            g_mean = torch.sigmoid(logits_gen).mean().item()

        return loss_td, float(e_mean), float(g_mean), 0.0

    @staticmethod
    def reward_from_logits(logits: torch.Tensor):
        return MILD2TransformerDiscriminator.reward_from_logits(logits)

    def train(self, epoch: int):
        self.P.train()
        for v in self.V:
            v.train()
        self.TD.train()

        gen_batch = self.P.collect_samples(
            self.config["ppo"]["sample_batch_size"],
            num_agents=self.config["general"]["num_agent"],
            make_env_fn=self.make_env_fn,
        )

        gen_state = multi_trans_shape_func(gen_batch.state)
        gen_action_raw = multi_trans_shape_func(gen_batch.action)
        gen_next_state = multi_trans_shape_func(gen_batch.next_state)
        gen_old_logp = multi_trans_shape_func(gen_batch.log_prob)
        gen_mask = multi_trans_shape_func(gen_batch.mask)

        gen_action = tuple(_ensure_onehot_action(a.to(device), self.action_n) for a in gen_action_raw)

        gen_obs_tok = torch.stack([gen_state[i].to(device) for i in range(self.num_agent)], dim=1)
        gen_act_tok = torch.stack([gen_action[i].to(device) for i in range(self.num_agent)], dim=1)

        d_steps = int(_get(self.config, "discriminator.d_steps", 1) or 1)
        K = int(_get(self.config, "discriminator.update_interval", 1) or 1)
        do_update_td = (epoch % K == 0)

        td_loss_list, e_list, g_list, gp_list = [], [], [], []

        if do_update_td:
            for _ in range(d_steps):
                expert_batch_state, expert_batch_action, _ = self._next_expert_batch()
                expert_batch_state = expert_batch_state.to(device)
                expert_batch_action = expert_batch_action.to(device)

                B = expert_batch_state.shape[0]
                exp_obs_tok = expert_batch_state.view(B, self.num_agent, self.obs_dim_per_agent)
                exp_act_tok = expert_batch_action.view(B, self.num_agent, self.action_n)

                loss_td, e_mean, g_mean, gp = self._disc_bce_loss(
                    exp_obs_tok, exp_act_tok, gen_obs_tok, gen_act_tok
                )

                self.optimizer_td.zero_grad(set_to_none=True)
                loss_td.backward()
                torch.nn.utils.clip_grad_norm_(self.TD.parameters(), 5.0)
                self.optimizer_td.step()

                td_loss_list.append(float(loss_td.item()))
                e_list.append(float(e_mean))
                g_list.append(float(g_mean))
                gp_list.append(float(gp))

            self.scheduler_td.step()
        else:
            td_loss_list.append(float("nan"))
            e_list.append(float("nan"))
            g_list.append(float("nan"))
            gp_list.append(0.0)

        with torch.no_grad():
            logits_gen = self.TD(gen_obs_tok, gen_act_tok)
            stat(logits_gen, "logits_gen")
            r_tok = self.reward_from_logits(logits_gen)
            gen_rewards = [r_tok[:, i:i+1].contiguous() for i in range(self.num_agent)]
            gen_rewards = [r.clamp(0.0, 10.0) for r in gen_rewards]
            print("[STAT] r_tok mean/std/min/max:",
                  r_tok.mean().item(), r_tok.std().item(),
                  r_tok.min().item(), r_tok.max().item())
        T = _as_int(self.config["jointpolicy"]["trajectory_length"], "jointpolicy.trajectory_length")
        adv_list, ret_list = [], []

        for i in range(self.num_agent):
            s_i = gen_state[i].to(device)
            v_i = self.V[i](s_i)
            adv_i, ret_i = gae_from_rewards(
                rewards=gen_rewards[i],
                values=v_i,
                masks=gen_mask[i].to(device),
                gamma=self._gamma,
                lam=self._tau,
                T=T,
            )
            adv_list.append(adv_i)
            ret_list.append(ret_i)

        ppo_optim_epochs = _as_int(self.config["ppo"]["ppo_optim_epochs"], "ppo.ppo_optim_epochs")
        ppo_mini_batch_size = _as_int(self.config["ppo"]["ppo_mini_batch_size"], "ppo.ppo_mini_batch_size")

        v_loss_list, p_loss_list = [], []

        for _ in range(ppo_optim_epochs):
            for agent_index in range(self.num_agent):
                batch_size = gen_state[agent_index].shape[0]
                optim_iter = int(math.ceil(batch_size / ppo_mini_batch_size))
                perm = torch.randperm(batch_size)

                for it in range(optim_iter):
                    ind = perm[slice(it * ppo_mini_batch_size,
                                     min((it + 1) * ppo_mini_batch_size, batch_size))]

                    mb_state = gen_state[agent_index][ind].to(device)
                    mb_action = gen_action[agent_index][ind].to(device)
                    mb_next = gen_next_state[agent_index][ind].to(device)
                    mb_adv = adv_list[agent_index][ind].to(device)
                    mb_ret = ret_list[agent_index][ind].to(device)
                    mb_oldlogp = gen_old_logp[agent_index][ind].to(device)

                    global_states = torch.cat([gen_state[i][ind].to(device) for i in range(self.num_agent)], dim=1)
                    global_actions = torch.cat([gen_action[i][ind].to(device) for i in range(self.num_agent)], dim=1)

                    v_loss, p_loss = ppo_step(
                        self.P, self.V[agent_index],
                        self.optimizer_policy,
                        self.optimizer_value[agent_index],
                        states=mb_state,
                        actions=mb_action,
                        next_states=mb_next,
                        returns=mb_ret,
                        old_log_probs=mb_oldlogp,
                        advantages=mb_adv,
                        ppo_clip_ratio=self._clip_ratio,
                        value_l2_reg=self._l2_reg,
                        index_agent=agent_index,
                        global_states=global_states,
                        global_actions=global_actions
                    )

                    v_loss_list.append(float(v_loss.item()))
                    p_loss_list.append(float(p_loss.item()))

        td_loss_mean = float(np.mean(td_loss_list))
        v_loss_mean = float(np.mean(v_loss_list)) if len(v_loss_list) else 0.0
        p_loss_mean = float(np.mean(p_loss_list)) if len(p_loss_list) else 0.0
        r_mean = float(torch.stack(gen_rewards).mean().item())

        self.writer.add_scalar("train/td/loss", td_loss_mean, epoch)
        self.writer.add_scalar("train/td/e_mean", float(np.mean(e_list)), epoch)
        self.writer.add_scalar("train/td/g_mean", float(np.mean(g_list)), epoch)
        self.writer.add_scalar("train/td/gp", float(np.mean(gp_list)), epoch)
        self.writer.add_scalar("train/ppo/p_loss", p_loss_mean, epoch)
        self.writer.add_scalar("train/ppo/v_loss", v_loss_mean, epoch)
        self.writer.add_scalar("train/reward/r_mean", r_mean, epoch)

        print(f"[MILD2] epoch={epoch} "
              f"td_bce={td_loss_mean:.4f} (D(exp)={np.mean(e_list):.3f}, D(gen)={np.mean(g_list):.3f}) "
              f"r={r_mean:.3f} p_loss={p_loss_mean:.4f} v_loss={v_loss_mean:.4f}")

        return td_loss_mean, v_loss_mean, p_loss_mean

    def eval(self, epoch: int):
        self.P.eval()
        for v in self.V:
            v.eval()
        self.TD.eval()

        with torch.no_grad():
            gen_batch = self.P.collect_samples(
                self.config["ppo"]["sample_batch_size"],
                num_agents=self.config["general"]["num_agent"],
                make_env_fn=self.make_env_fn,
            )
            gen_state = multi_trans_shape_func(gen_batch.state)
            gen_action_raw = multi_trans_shape_func(gen_batch.action)
            gen_action = tuple(_ensure_onehot_action(a.to(device), self.action_n) for a in gen_action_raw)

            gen_obs_tok = torch.stack([gen_state[i].to(device) for i in range(self.num_agent)], dim=1)
            gen_act_tok = torch.stack([gen_action[i].to(device) for i in range(self.num_agent)], dim=1)

            logits = self.TD(gen_obs_tok, gen_act_tok)
            rewards = self.reward_from_logits(logits)
            rewards = rewards.clamp(0.0, 10.0)
            score = float(rewards.mean().item())

            self.writer.add_scalar("validate/reward/mean", score, epoch)
            print(f"[MILD2 eval] epoch={epoch} reward_mean={score:.4f}")

    def save_model(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.P,  f"{save_path}/{self.exp_name}_JointPolicy.pt")
        torch.save(self.V,  f"{save_path}/{self.exp_name}_Value.pt")
        torch.save(self.TD, f"{save_path}/{self.exp_name}_TransformerDiscriminator.pt")
