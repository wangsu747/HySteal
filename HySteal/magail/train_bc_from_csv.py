


import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.il_csv_adapter import normalize_il_dataframe

@dataclass
class AugmentConfig:
    enable: bool = False
    p_aug: float = 1.0
    gauss_sigma: float = 0.01
    p_mask: float = 0.0
    mask_value: float = 0.0
    mask_mode: str = "feature"

    alpha_ce: float = 0.0

    lambda_cons: float = 0.1
    cons_type: str = "kl_probs"



def _parse_dims_spec(spec: str, max_dim: int) -> torch.Tensor:
    mask = torch.zeros(max_dim, dtype=torch.bool)
    if spec is None or len(spec.strip()) == 0:
        return mask
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            lo, hi = int(a), int(b)
            lo = max(0, lo); hi = min(max_dim - 1, hi)
            if lo <= hi:
                mask[lo:hi + 1] = True
        else:
            d = int(p)
            if 0 <= d < max_dim:
                mask[d] = True
    return mask


def augment_obs_batch(
    obs: torch.Tensor,
    cfg: AugmentConfig,
    global_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if (not cfg.enable) or (cfg.p_aug <= 0.0):
        return obs

    if torch.rand(()) > cfg.p_aug:
        return obs

    x = obs

    if cfg.gauss_sigma and cfg.gauss_sigma > 0:
        if global_mask is None:
            noise = torch.randn_like(x) * cfg.gauss_sigma
        else:
            noise = torch.randn_like(x) * cfg.gauss_sigma
            if global_mask.any():
                shared = (torch.randn(1, x.shape[1], device=x.device, dtype=x.dtype) * cfg.gauss_sigma)
                noise[:, global_mask] = shared[:, global_mask]
        x = x + noise

    if cfg.p_mask and cfg.p_mask > 0:
        if cfg.mask_mode == "feature":
            m_feat = (torch.rand(1, x.shape[1], device=x.device) < cfg.p_mask)
            x = torch.where(m_feat, torch.full_like(x, cfg.mask_value), x)
        elif cfg.mask_mode == "element":
            m = (torch.rand_like(x) < cfg.p_mask)
            x = torch.where(m, torch.full_like(x, cfg.mask_value), x)
        else:
            raise ValueError(f"Unknown mask_mode: {cfg.mask_mode}")

    return x


def consistency_loss(
    logits_clean: torch.Tensor,
    logits_aug: torch.Tensor,
    cfg: AugmentConfig,
) -> torch.Tensor:
    if cfg.lambda_cons <= 0:
        return torch.tensor(0.0, device=logits_clean.device, dtype=logits_clean.dtype)

    if cfg.cons_type == "kl_probs":
        p = F.softmax(logits_clean, dim=-1).detach()
        logq = F.log_softmax(logits_aug, dim=-1)
        return F.kl_div(logq, p, reduction="batchmean")
    elif cfg.cons_type == "mse_logits":
        return F.mse_loss(logits_clean.detach(), logits_aug)
    else:
        raise ValueError(f"Unknown cons_type: {cfg.cons_type}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infer_obs_cols(df: pd.DataFrame) -> List[str]:
    obs_cols = [c for c in df.columns if c.startswith("obs_") and c[len("obs_"):].isdigit()]
    obs_cols = sorted(obs_cols, key=lambda x: int(x.split("_")[1]))
    if len(obs_cols) == 0:
        raise ValueError("No obs_* columns found in CSV.")
    return obs_cols


def infer_action_dim(df: pd.DataFrame, action_col: str = "action_id") -> int:
    if action_col not in df.columns:
        raise ValueError(f"Missing required column: {action_col}")
    max_a = int(df[action_col].max())
    if max_a < 0:
        raise ValueError("action_id max < 0, invalid.")
    return max_a + 1


def infer_agent_order(df: pd.DataFrame, agent_order_arg: Optional[str]) -> List[str]:
    if agent_order_arg and len(agent_order_arg.strip()) > 0:
        order = [x.strip() for x in agent_order_arg.split(",") if x.strip()]
        if len(order) == 0:
            raise ValueError("agent_order is empty after parsing.")
        return order

    if "agent" not in df.columns:
        raise ValueError("Missing required column: agent")
    seen = []
    s = set()
    for a in df["agent"].tolist():
        if a not in s:
            seen.append(a)
            s.add(a)
    return seen


@dataclass
class MLPConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int
    n_layers: int
    use_agent_id: bool
    agent_emb_dim: int
    n_agents: int

class BCPolicy(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        in_dim = cfg.obs_dim
        if cfg.use_agent_id:
            self.agent_emb = nn.Embedding(cfg.n_agents, cfg.agent_emb_dim)
            in_dim += cfg.agent_emb_dim
        else:
            self.agent_emb = None

        layers = []
        cur = in_dim
        for _ in range(cfg.n_layers):
            layers.append(nn.Linear(cur, cfg.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            cur = cfg.hidden_dim
        layers.append(nn.Linear(cur, cfg.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, agent_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.use_agent_id:
            if agent_id is None:
                raise ValueError("agent_id is required when use_agent_id=True")
            emb = self.agent_emb(agent_id)
            x = torch.cat([obs, emb], dim=-1)
        else:
            x = obs
        return self.net(x)


def make_splits_by_episode(df: pd.DataFrame, seed: int, val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    if "episode" not in df.columns:
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = int(len(df) * val_ratio)
        return idx[n_val:], idx[:n_val]

    eps = df["episode"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val_eps = max(1, int(len(eps) * val_ratio)) if len(eps) > 1 else 0
    val_eps = set(eps[:n_val_eps])
    is_val = df["episode"].isin(val_eps).to_numpy()
    val_idx = np.where(is_val)[0]
    train_idx = np.where(~is_val)[0]
    if len(val_idx) == 0:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_val = max(1, int(len(df) * val_ratio))
        return idx[n_val:], idx[:n_val]
    return train_idx, val_idx


def batch_iter(indices: np.ndarray, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i:i + batch_size]


@torch.no_grad()
def evaluate(model: BCPolicy,
             obs: torch.Tensor,
             agent_ids: Optional[torch.Tensor],
             actions: torch.Tensor,
             batch_size: int,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    n = obs.shape[0]
    for i in range(0, n, batch_size):
        o = obs[i:i + batch_size].to(device)
        a = actions[i:i + batch_size].to(device)
        if agent_ids is not None:
            ag = agent_ids[i:i + batch_size].to(device)
            logits = model(o, ag)
        else:
            logits = model(o, None)
        loss = F.cross_entropy(logits, a, reduction="sum")
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_correct += int((pred == a).sum().item())
        total += int(a.numel())
    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--use_agent_id", action="store_true")
    p.add_argument("--agent_order", type=str, default="", help="comma-separated, e.g. adversary_0,adversary_1,agent_0")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--agent_emb_dim", type=int, default=16)

    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--out_dir", type=str, default=".")

    p.add_argument("--aug_enable", action="store_true", help="enable online observation augmentation")
    p.add_argument("--aug_p", type=float, default=1.0)
    p.add_argument("--aug_sigma", type=float, default=0.01)
    p.add_argument("--aug_p_mask", type=float, default=0.0)
    p.add_argument("--mask_mode", type=str, default="feature", choices=["feature", "element"])
    p.add_argument("--alpha_ce", type=float, default=0.0)
    p.add_argument("--lambda_cons", type=float, default=0.1)
    p.add_argument("--cons_type", type=str, default="kl_probs", choices=["mse_logits", "kl_probs"])
    p.add_argument("--aug_mask_value", type=float, default=0.0)

    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    df = pd.read_csv(args.csv_path, low_memory=False)
    df, _ = normalize_il_dataframe(df, obs_dim=None, fill_value=0.0, ensure_next_obs=False)
    obs_cols = infer_obs_cols(df)
    obs_dim = len(obs_cols)
    nan_cnt = int(df[obs_cols].isna().sum().sum())
    if nan_cnt > 0:
        print(f"[warn] found NaNs in obs_*: {nan_cnt}. Filling with 0.0")
        df[obs_cols] = df[obs_cols].fillna(0.0)

    action_dim = infer_action_dim(df, "action_id")
    agent_order = infer_agent_order(df, args.agent_order)
    agent_to_id = {a: i for i, a in enumerate(agent_order)}
    n_agents = len(agent_order)

    if args.use_agent_id:
        known_mask = df["agent"].isin(agent_to_id.keys())
        if not bool(known_mask.all()):
            df = df.loc[known_mask].copy()

    obs_np = df[obs_cols].to_numpy(dtype=np.float32)
    act_np = df["action_id"].to_numpy(dtype=np.int64)
    if args.use_agent_id:
        ag_np = df["agent"].map(agent_to_id).to_numpy(dtype=np.int64)
    else:
        ag_np = None

    train_idx, val_idx = make_splits_by_episode(df, args.seed, args.val_ratio)

    obs_t = torch.from_numpy(obs_np)
    act_t = torch.from_numpy(act_np)
    ag_t = torch.from_numpy(ag_np) if ag_np is not None else None

    cfg = MLPConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        use_agent_id=args.use_agent_id,
        agent_emb_dim=args.agent_emb_dim,
        n_agents=n_agents,
    )
    model = BCPolicy(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    aug_cfg = AugmentConfig(
        enable=bool(args.aug_enable),
        p_aug=float(args.aug_p),
        gauss_sigma=float(args.aug_sigma),
        p_mask=float(args.aug_p_mask),
        mask_value=float(args.aug_mask_value),
        mask_mode=str(args.mask_mode),
        alpha_ce=float(args.alpha_ce),
        lambda_cons=float(args.lambda_cons),
        cons_type=str(args.cons_type),
    )

    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, f"bc_best_seed{args.seed}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for bidx in batch_iter(train_idx, args.batch_size, seed=args.seed + epoch * 1000):
            o = obs_t[bidx].to(device)
            a = act_t[bidx].to(device)
            o_aug = augment_obs_batch(o, aug_cfg)

            if cfg.use_agent_id:
                ag = ag_t[bidx].to(device)
                logits = model(o, ag)
                logits_aug = model(o_aug, ag)
            else:
                logits = model(o, None)
                logits_aug = model(o_aug, None)

            loss_clean = F.cross_entropy(logits, a)

            loss = loss_clean

            if aug_cfg.alpha_ce and aug_cfg.alpha_ce > 0:
                loss_aug = F.cross_entropy(logits_aug, a)
                loss = (1.0 - aug_cfg.alpha_ce) * loss_clean + aug_cfg.alpha_ce * loss_aug

            loss_cons = consistency_loss(logits, logits_aug, aug_cfg)
            loss = loss + aug_cfg.lambda_cons * loss_cons

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = int(a.numel())
            total_loss += float(loss.item()) * bs
            total += bs

        train_loss = total_loss / max(1, total)
        val_metrics = evaluate(
            model,
            obs_t[val_idx],
            ag_t[val_idx] if ag_t is not None else None,
            act_t[val_idx],
            batch_size=max(256, args.batch_size // 4),
            device=device,
        )
        if epoch == 1:
            print(f"[aug] enable={aug_cfg.enable} p={aug_cfg.p_aug} sigma={aug_cfg.gauss_sigma} "
                  f"p_mask={aug_cfg.p_mask} mask_mode={aug_cfg.mask_mode} alpha_ce={aug_cfg.alpha_ce} "
                  f"lambda_cons={aug_cfg.lambda_cons} cons_type={aug_cfg.cons_type}")

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.3f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt = {
                "model_state_dict": model.state_dict(),
                "obs_dim": cfg.obs_dim,
                "action_dim": cfg.action_dim,
                "agent_order": agent_order,
                "use_agent_id": cfg.use_agent_id,
                "hidden_dim": cfg.hidden_dim,
                "n_layers": cfg.n_layers,
                "agent_emb_dim": cfg.agent_emb_dim,
            }
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(ckpt, best_path)
            print(f"  -> saved best to {best_path}")

    print(f"Done. Best val_loss={best_val_loss:.4f} ckpt={best_path}")


if __name__ == "__main__":
    main()
