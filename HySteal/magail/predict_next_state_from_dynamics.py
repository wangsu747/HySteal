


import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from magail.dynamics_env import ResidualDynamicsMLP, DynCfg


def to_vec(s: str, expected_dim: int):
    arr = np.asarray(json.loads(s), dtype=np.float32).reshape(-1)
    if arr.shape[0] == expected_dim:
        return arr
    if arr.shape[0] > expected_dim:
        return arr[:expected_dim]
    out = np.zeros((expected_dim,), dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out

def main():
    p = argparse.ArgumentParser("Predict joint next-state from trained dynamics model")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--joint_state_json", type=str, required=True, help="JSON list, length=joint_state_dim")
    p.add_argument("--joint_action_json", type=str, required=True, help="JSON list, length=joint_action_dim")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(args.ckpt_path, map_location=device)

    model = ResidualDynamicsMLP(
        DynCfg(
            in_dim=int(ckpt["joint_state_dim"] + ckpt["joint_action_dim"]),
            out_dim=int(ckpt["joint_state_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            n_layers=int(ckpt["n_layers"]),
        )
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    s = to_vec(args.joint_state_json, int(ckpt["joint_state_dim"]))
    a = to_vec(args.joint_action_json, int(ckpt["joint_action_dim"]))
    st = torch.from_numpy(s).unsqueeze(0).to(device)
    at = torch.from_numpy(a).unsqueeze(0).to(device)
    with torch.no_grad():
        y, _ = model(st, at)
    out = y.squeeze(0).detach().cpu().numpy().tolist()
    print(json.dumps({"pred_next_joint_state": out}))


if __name__ == "__main__":
    main()
