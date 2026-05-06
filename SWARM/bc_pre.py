import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.utils.il_csv_adapter import normalize_il_dataframe


# dataset
class ExpertCSVDatasetPerAgent(Dataset):
    """
    Each row = one agent timestep.
    Use:
      - obs_0..obs_{per_agent_obs_dim-1}
      - agent -> agent_id via agent_order
      - label: action_id
      - optional: weight
    """
    def __init__(self, df, per_agent_obs_dim=16, agent_order=None):
        if agent_order is None:
            agent_order = sorted(df["agent"].unique().tolist())

        self.agent_order = list(agent_order)
        self.agent_to_id = {a: i for i, a in enumerate(self.agent_order)}

        obs_cols = [f"obs_{i}" for i in range(per_agent_obs_dim)]
        need = obs_cols + ["agent", "action_id"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"[BC] missing columns: {miss}")

        # filter invalid agents
        df = df[df["agent"].isin(self.agent_order)].copy()

        self.obs = torch.tensor(df[obs_cols].to_numpy(np.float32))
        self.act_id = torch.tensor(df["action_id"].to_numpy(np.int64))
        self.agent_id = torch.tensor(df["agent"].map(self.agent_to_id).to_numpy(np.int64))

        if "weight" in df.columns:
            self.w = torch.tensor(df["weight"].to_numpy(np.float32))
        else:
            self.w = torch.ones(len(df), dtype=torch.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, i):
        return {
            "obs": self.obs[i],            # [16]
            "agent_id": self.agent_id[i],  # scalar
            "act_id": self.act_id[i],      # scalar (0..action_n-1)
            "w": self.w[i],                # scalar
        }


#  build global_state
def build_global_state_slot(states: torch.Tensor, agent_id: int, num_agent: int, per_agent_dim: int) -> torch.Tensor:
    """
    Build global_state [B, num_agent*per_agent_dim] by putting this agent's states
    into its slot, and zeros elsewhere.

    This matches JointPolicy.collect_samples global_state shape.
    """
    B = states.size(0)
    dev = states.device
    global_states = torch.zeros((B, num_agent * per_agent_dim), device=dev, dtype=states.dtype)
    s = agent_id * per_agent_dim
    e = s + per_agent_dim
    global_states[:, s:e] = states
    return global_states


def onehot_actions(act_id: torch.Tensor, action_n: int) -> torch.Tensor:
    """
    act_id: [B] long
    return: [B, action_n] float onehot
    """
    return F.one_hot(act_id.long(), num_classes=action_n).to(dtype=torch.float32)


# adapter for JointPolicy
def get_log_prob_jointpolicy(policy, states, act_id, agent_id_int, num_agent, per_agent_dim, action_n):
    """
    policy: JointPolicy
    states: [B, 16]
    act_id: [B] long
    agent_id_int: python int
    """
    # 1) build global_state [B,64] (slot scatter)
    global_states = build_global_state_slot(states, agent_id_int, num_agent, per_agent_dim)

    # 2) actions as onehot float [B,5]  (VERY likely what Actor.get_log_prob expects)
    actions = onehot_actions(act_id, action_n).to(device=states.device, dtype=states.dtype)

    # 3) call exactly signature: (states, actions, index_agent, global_states)
    out = policy.get_log_prob(states, actions, agent_id_int, global_states)

    # out might be [B,1] or [B]
    if torch.is_tensor(out):
        return out.view(-1)

    # sometimes dict
    if isinstance(out, dict):
        for k in ["logp", "log_prob", "action_logp", "log_probs"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k].view(-1)

    return None


@torch.no_grad()
def bc_eval(policy, loader, device, num_agent, per_agent_dim, action_n):
    policy.eval()
    total = 0
    loss_sum = 0.0

    for batch in loader:
        obs = batch["obs"].to(device)          # [B,16]
        agent_id = batch["agent_id"].to(device)  # [B]
        act_id = batch["act_id"].to(device)    # [B]
        w = batch["w"].to(device)              # [B] or scalar

        # index_agent must be int
        unique_aids = agent_id.unique().tolist()
        for aid in unique_aids:
            mask = (agent_id == aid)
            if mask.sum().item() == 0:
                continue
            obs_i = obs[mask]
            act_i = act_id[mask]
            w_i = w[mask]

            logp = get_log_prob_jointpolicy(policy, obs_i, act_i, int(aid), num_agent, per_agent_dim, action_n)
            if logp is None:
                raise RuntimeError("[BC] cannot get log_prob in eval. Check Actor.get_log_prob output type/shape.")

            loss_i = -logp
            loss = (loss_i * w_i).mean()

            bs = act_i.numel()
            loss_sum += loss.item() * bs
            total += bs

    return loss_sum / max(1, total)


# main bc
def bc_pretrain(
    policy,
    expert_csv_path,
    device="cuda",
    per_agent_obs_dim=16,
    num_agent=4,
    action_n=5,
    agent_order=None,
    epochs=200,
    batch_size=1000,
    lr=3e-4,
    weight_decay=0.0,
    grad_clip=1.0,
    val_ratio=0.1,
    seed=0,
    save_path=None,
    num_workers=4,
):
    """
    BC pretrain for JointPolicy:
    """
    rng = np.random.RandomState(seed)

    df = pd.read_csv(expert_csv_path, low_memory=False)
    df, _ = normalize_il_dataframe(df, obs_dim=per_agent_obs_dim, fill_value=0.0, ensure_next_obs=False)

    if "source" in df.columns:
        df = df[df["source"] == "expert"].copy()

    if "episode" not in df.columns:
        raise ValueError("[BC] CSV missing 'episode' column for episode-level split.")

    episodes = sorted(df["episode"].unique().tolist())
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_ratio))
    val_eps = set(episodes[:n_val])
    train_eps = set(episodes[n_val:])

    train_df = df[df["episode"].isin(list(train_eps))].copy()
    val_df = df[df["episode"].isin(list(val_eps))].copy()

    train_ds = ExpertCSVDatasetPerAgent(train_df, per_agent_obs_dim=per_agent_obs_dim, agent_order=agent_order)
    val_ds = ExpertCSVDatasetPerAgent(val_df, per_agent_obs_dim=per_agent_obs_dim, agent_order=agent_order)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True
    )

    policy = policy.to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    if save_path is None:
        save_path = os.path.join(os.getcwd(), "bc_pretrained_policy.pt")

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        policy.train()

        for batch in train_loader:
            obs = batch["obs"].to(device)              # [B,16]
            agent_id = batch["agent_id"].to(device)    # [B]
            act_id = batch["act_id"].to(device)        # [B]
            w = batch["w"].to(device)                  # [B]

            # per-agent groups (because index_agent must be int)
            unique_aids = agent_id.unique().tolist()
            losses = []

            for aid in unique_aids:
                mask = (agent_id == aid)
                if mask.sum().item() == 0:
                    continue

                obs_i = obs[mask]
                act_i = act_id[mask]
                w_i = w[mask]

                logp = get_log_prob_jointpolicy(policy, obs_i, act_i, int(aid), num_agent, per_agent_obs_dim, action_n)
                if logp is None:
                    raise RuntimeError(
                        "[BC] still cannot get log_prob.\n"
                        "Now we are calling: get_log_prob(states, onehot_actions, index_agent:int, global_states[slot]).\n"
                        "If this fails, Actor.get_log_prob likely expects different action format (id vs onehot) "
                        "or different global_states composition."
                    )

                loss_i = (-logp)  # [b_i]
                losses.append((loss_i * w_i).mean())

            if len(losses) == 0:
                continue

            loss = torch.stack(losses).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            opt.step()

        val_loss = bc_eval(policy, val_loader, device, num_agent, per_agent_obs_dim, action_n)
        print(f"[BC] {ep:03d}/{epochs}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "policy_state_dict": policy.state_dict(),
                "agent_to_id": train_ds.agent_to_id,
                "per_agent_obs_dim": per_agent_obs_dim,
                "num_agent": num_agent,
                "action_n": action_n,
                "agent_order": agent_order,
            }, save_path)
            print(f"[BC] saved best -> {save_path}")

    return save_path
