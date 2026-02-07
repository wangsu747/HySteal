


import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.il_csv_adapter import normalize_il_dataframe, infer_obs_cols
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def infer_agent_order(df: pd.DataFrame, agent_order_arg: str) -> List[str]:
    if isinstance(agent_order_arg, str) and agent_order_arg.strip():
        return [x.strip() for x in agent_order_arg.split(",") if x.strip()]
    if "agent" not in df.columns:
        raise ValueError("CSV missing `agent`.")
    seen = []
    s = set()
    for a in df["agent"].tolist():
        if a not in s:
            s.add(a)
            seen.append(str(a))
    return seen


def _filter_complete_group(g, agent_order, require_full: bool):
    if require_full and len(g) != len(agent_order):
        return None
    if g["agent"].nunique() != len(g):
        return None
    if require_full and list(g["agent"].values) != agent_order:
        return None
    return g


def build_joint_transitions(
    df: pd.DataFrame,
    agent_order: List[str],
    obs_dim: int,
    action_n: int,
    require_full: bool = True,
):
    agent_to_idx = {a: i for i, a in enumerate(agent_order)}
    n_agents = len(agent_order)

    keep_cols = (
        ["episode", "t", "agent", "action_id"]
        + [f"obs_{i}" for i in range(obs_dim)]
        + [f"next_obs_{i}" for i in range(obs_dim)]
    )
    for c in keep_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column after normalization: {c}")
    d = df[keep_cols].copy()
    d = d[d["agent"].isin(agent_order)].copy()

    xs, us, ys, eps = [], [], [], []
    for (ep, _t), g in d.groupby(["episode", "t"], sort=True):
        g = g.sort_values(by="agent", key=lambda s: s.map(agent_to_idx))
        g = _filter_complete_group(g, agent_order, require_full)
        if g is None:
            continue

        obs = g[[f"obs_{i}" for i in range(obs_dim)]].to_numpy(dtype=np.float32)
        nxt = g[[f"next_obs_{i}" for i in range(obs_dim)]].to_numpy(dtype=np.float32)
        act = g["action_id"].to_numpy(dtype=np.int64)

        act_oh = np.zeros((n_agents, action_n), dtype=np.float32)
        valid = (act >= 0) & (act < action_n)
        act_oh[np.arange(n_agents)[valid], act[valid]] = 1.0

        xs.append(obs.reshape(-1))
        us.append(act_oh.reshape(-1))
        ys.append(nxt.reshape(-1))
        eps.append(int(ep))

    if len(xs) == 0:
        raise RuntimeError("No valid joint transitions found.")
    return (
        np.stack(xs).astype(np.float32),
        np.stack(us).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.asarray(eps, dtype=np.int64),
    )

def build_init_states(df: pd.DataFrame, agent_order: List[str], obs_dim: int, require_full: bool = True):
    agent_to_idx = {a: i for i, a in enumerate(agent_order)}
    keep_cols = ["episode", "t", "agent"] + [f"obs_{i}" for i in range(obs_dim)]
    for c in keep_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column after normalization: {c}")
    d = df[keep_cols].copy()
    d = d[d["agent"].isin(agent_order)].copy()

    inits = []
    for (ep, t), g in d.groupby(["episode", "t"], sort=True):
        if int(t) != 0:
            continue
        g = g.sort_values(by="agent", key=lambda s: s.map(agent_to_idx))
        g = _filter_complete_group(g, agent_order, require_full)
        if g is None:
            continue
        obs = g[[f"obs_{i}" for i in range(obs_dim)]].to_numpy(dtype=np.float32)
        inits.append(obs.reshape(-1))

    return np.stack(inits).astype(np.float32) if len(inits) else np.zeros((1, obs_dim * len(agent_order)), dtype=np.float32)


def split_by_episode(ep_ids: np.ndarray, seed: int, val_ratio: float):
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_ratio)) if len(uniq) > 1 else 0
    val_set = set(uniq[:n_val].tolist())
    is_val = np.asarray([e in val_set for e in ep_ids], dtype=bool)
    val_idx = np.where(is_val)[0]
    tr_idx = np.where(~is_val)[0]
    if len(val_idx) == 0:
        idx = np.arange(len(ep_ids))
        rng.shuffle(idx)
        nv = max(1, int(len(idx) * val_ratio))
        return idx[nv:], idx[:nv]
    return tr_idx, val_idx


@dataclass
class DynCfg:
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int


class ResidualDynamicsMLP(nn.Module):
    def __init__(self, cfg: DynCfg):
        super().__init__()
        layers = []
        cur = cfg.in_dim
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(cur, cfg.hidden_dim), nn.ReLU(inplace=True)]
            cur = cfg.hidden_dim
        layers += [nn.Linear(cur, cfg.out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        delta = self.net(x)
        return s + delta, delta


@torch.no_grad()
def evaluate(model, s, a, y, idx, bs, device):
    model.eval()
    n = len(idx)
    mse_sum = 0.0
    mae_sum = 0.0
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        sb = s[b].to(device)
        ab = a[b].to(device)
        yb = y[b].to(device)
        yp, _ = model(sb, ab)
        mse_sum += float(F.mse_loss(yp, yb, reduction="sum").item())
        mae_sum += float(F.l1_loss(yp, yb, reduction="sum").item())
    denom = max(1, n * y.shape[1])
    return {"mse": mse_sum / denom, "mae": mae_sum / denom}


def main():
    p = argparse.ArgumentParser("Train offline multi-agent dynamics from trajectory CSV")
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--agent_order", type=str, default="")
    p.add_argument("--obs_dim_per_agent", type=int, default=0, help="0 means infer from CSV")
    p.add_argument("--action_n", type=int, default=0, help="0 means infer from CSV")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_path", type=str, default="hysteal_ckpts/dynamics_best.pt")
    p.add_argument("--max_init_states", type=int, default=10000)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    df = pd.read_csv(args.csv_path, low_memory=False)
    df, inferred_obs_dim = normalize_il_dataframe(df, obs_dim=None, fill_value=0.0, ensure_next_obs=True)
    obs_cols = infer_obs_cols(df)
    if len(obs_cols) == 0:
        raise ValueError("No obs_* columns found after normalization.")
    obs_dim = int(args.obs_dim_per_agent) if int(args.obs_dim_per_agent) > 0 else int(inferred_obs_dim)
    if obs_dim <= 0:
        obs_dim = len(obs_cols)
    action_n = int(args.action_n) if int(args.action_n) > 0 else (int(pd.to_numeric(df["action_id"], errors="coerce").fillna(0).max()) + 1)
    agent_order = infer_agent_order(df, args.agent_order)
    n_agents = len(agent_order)

    x_np, u_np, y_np, ep_np = build_joint_transitions(
        df=df,
        agent_order=agent_order,
        obs_dim=obs_dim,
        action_n=action_n,
        require_full=True,
    )

    init_states = build_init_states(df, agent_order, obs_dim, require_full=True)
    if init_states.shape[0] > int(args.max_init_states):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(init_states.shape[0], size=int(args.max_init_states), replace=False)
        init_states = init_states[idx]

    tr_idx, va_idx = split_by_episode(ep_np, args.seed, args.val_ratio)

    s = torch.from_numpy(x_np)
    a = torch.from_numpy(u_np)
    y = torch.from_numpy(y_np)

    model = ResidualDynamicsMLP(
        DynCfg(
            in_dim=x_np.shape[1] + u_np.shape[1],
            out_dim=y_np.shape[1],
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        )
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mse = float("inf")
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        idx = tr_idx.copy()
        np.random.default_rng(args.seed + ep).shuffle(idx)
        loss_sum = 0.0
        n_steps = 0
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            sb = s[b].to(device)
            ab = a[b].to(device)
            yb = y[b].to(device)

            yp, _ = model(sb, ab)
            loss = F.smooth_l1_loss(yp, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            loss_sum += float(loss.item())
            n_steps += 1

        va = evaluate(model, s, a, y, va_idx, bs=max(256, args.batch_size // 4), device=device)
        print(f"[Dyn] ep={ep:03d}/{args.epochs} train_huber={loss_sum/max(1,n_steps):.6f} val_mse={va['mse']:.6f} val_mae={va['mae']:.6f}")

        if va["mse"] < best_mse:
            best_mse = va["mse"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "agent_order": agent_order,
                "n_agents": n_agents,
                "obs_dim_per_agent": obs_dim,
                "action_n": action_n,
                "joint_state_dim": int(x_np.shape[1]),
                "joint_action_dim": int(u_np.shape[1]),
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "best_val_mse": best_mse,
                "csv_path": args.csv_path,
                "init_joint_states": init_states.astype(np.float32),
            }
            torch.save(ckpt, args.out_path)
            print(f"  -> saved best to {args.out_path}")

    print(f"Done. best_val_mse={best_mse:.6f}, ckpt={args.out_path}")


if __name__ == "__main__":
    main()
