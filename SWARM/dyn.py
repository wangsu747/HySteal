import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.common import agent_order_csv, ensure_dir, python_exe, repo_root, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SWARM dyn")
    p.add_argument("--env", choices=["simple_tag_v3", "simple_spread_v3", "hvac"], required=True)
    p.add_argument("--csv_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--agent_order", default="")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(os.path.dirname(args.out_path) or ".")
    cmd = [
        python_exe(),
        os.path.join(repo_root(), "SWARM", "dyn_csv.py"),
        "--csv_path", args.csv_path,
        "--agent_order", agent_order_csv(args.env, args.agent_order),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--device", args.device,
        "--out_path", args.out_path,
    ]
    run(cmd)


if __name__ == "__main__":
    main()
