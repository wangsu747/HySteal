import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.common import agent_order_csv, ensure_dir, python_exe, repo_root, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SWARM weak")
    p.add_argument("--env", choices=["simple_tag_v3", "simple_spread_v3", "hvac"], required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--agent_order", default="")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1028)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--seed_end", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_agent_id", action="store_true", default=True)
    p.add_argument("--aug_p", type=float, default=1.0)
    p.add_argument("--sigma_base", type=float, default=0.01)
    p.add_argument("--sigma_step", type=float, default=0.005)
    p.add_argument("--alpha_ce", type=float, default=0.0)
    p.add_argument("--lambda_cons", type=float, default=0.3)
    p.add_argument("--cons_type", default="kl_probs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    order = agent_order_csv(args.env, args.agent_order)
    for seed in range(args.seed_start, args.seed_end + 1):
        sigma = args.sigma_base + args.sigma_step * seed
        cmd = [
            python_exe(),
            os.path.join(repo_root(), "SWARM", "bc_csv.py"),
            "--csv_path", args.csv_path,
            "--agent_order", order,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(seed),
            "--device", args.device,
            "--out_dir", args.out_dir,
            "--aug_enable",
            "--aug_p", str(args.aug_p),
            "--aug_sigma", str(sigma),
            "--alpha_ce", str(args.alpha_ce),
            "--lambda_cons", str(args.lambda_cons),
            "--cons_type", args.cons_type,
        ]
        if args.use_agent_id:
            cmd.append("--use_agent_id")
        run(cmd)


if __name__ == "__main__":
    main()
