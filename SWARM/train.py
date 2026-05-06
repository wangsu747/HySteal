import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.common import ensure_dir, num_agent, python_exe, repo_root, run


def _count_episodes(csv_path: str) -> int:
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    if "episode" not in df.columns:
        raise ValueError(f"missing episode column: {csv_path}")
    return int(pd.to_numeric(df["episode"], errors="coerce").dropna().astype(int).nunique())


def _maybe_convert_hvac_legacy(csv_path: str, out_dir: str) -> str:
    if csv_path.endswith("_legacy.csv"):
        return csv_path
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "train_input_legacy.csv")
    cmd = [
        python_exe(),
        os.path.join(repo_root(), "SWARM", "hvac", "to_legacy.py"),
        "--in_csv", csv_path,
        "--out_csv", out_csv,
    ]
    run(cmd)
    return out_csv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SWARM train")
    p.add_argument("--env", choices=["simple_tag_v3", "simple_spread_v3", "hvac"], required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--dynamics_ckpt", default="")
    p.add_argument("--trial", type=int, default=95005)
    p.add_argument("--eval_model_epoch", type=int, default=2)
    p.add_argument("--save_model_epoch", type=int, default=50)
    p.add_argument("--bc_pretrain_epochs", type=int, default=50)
    p.add_argument("--bc_batch_size", type=int, default=1000)
    p.add_argument("--bc_lr", type=float, default=3e-4)
    p.add_argument("--reset_from_t0_only", action="store_true")
    p.add_argument("--reset_noise_sigma", type=float, default=0.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--subset_size", type=int, default=None)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--expert_policy_ckpt", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    if args.env == "hvac":
        csv_path = _maybe_convert_hvac_legacy(args.csv_path, args.out_dir)
        subset_size = args.subset_size if args.subset_size is not None else _count_episodes(csv_path)
        cmd = [
            python_exe(),
            os.path.join(repo_root(), "SWARM", "hvac", "sweep_hvac.py"),
            "--csv_path", csv_path,
            "--config", args.config or os.path.join(repo_root(), "SWARM", "configs", "campus_4agent.yml"),
            "--subset_sizes", str(subset_size),
            "--horizon", str(args.horizon),
            "--eval_episodes", str(args.eval_episodes),
            "--expert_policy_ckpt", args.expert_policy_ckpt,
            "--train_bc",
            "--train_magail",
            "--out_dir", args.out_dir,
        ]
        run(cmd)
        return

    model_dir = os.path.join(args.out_dir, "models")
    log_dir = os.path.join(args.out_dir, "logs")
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    cmd = [
        python_exe(),
        os.path.join(repo_root(), "SWARM", "magail_train.py"),
        "--train_mode", "True",
        "--config_train", args.config or os.path.join(repo_root(), "SWARM", "configs", "config_multi.yml"),
        "--expert_csv", args.csv_path,
        "--dynamics_ckpt", args.dynamics_ckpt,
        "--num_agent", str(num_agent(args.env)),
        "--trial", str(args.trial),
        "--eval_model_epoch", str(args.eval_model_epoch),
        "--save_model_epoch", str(args.save_model_epoch),
        "--save_model_path", model_dir,
        "--log_root", log_dir,
        "--bc_pretrain_epochs", str(args.bc_pretrain_epochs),
        "--bc_batch_size", str(args.bc_batch_size),
        "--bc_lr", str(args.bc_lr),
        "--reset_from_t0_only", "True" if args.reset_from_t0_only else "False",
        "--reset_noise_sigma", str(args.reset_noise_sigma),
    ]
    if args.plot:
        cmd.append("--plot")
    run(cmd)


if __name__ == "__main__":
    main()
