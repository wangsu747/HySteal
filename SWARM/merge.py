import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.common import ensure_dir, max_cycles, python_exe, repo_root, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SWARM merge")
    p.add_argument("--env", choices=["simple_tag_v3", "simple_spread_v3", "hvac"], default="")
    p.add_argument("--expert_csv", required=True)
    p.add_argument("--aug_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--expert_weight", type=float, default=1.0)
    p.add_argument("--aug_weight", type=float, default=0.5)
    p.add_argument("--fill_nan_obs", type=float, default=0.0)
    p.add_argument("--max_t", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    max_t = int(args.max_t)
    if max_t <= 0:
        if not args.env:
            raise ValueError("provide --env or --max_t")
        max_t = int(max_cycles(args.env))
    cmd = [
        python_exe(),
        os.path.join(repo_root(), "SWARM", "merge_csv.py"),
        "--expert_csv", args.expert_csv,
        "--aug_csv", args.aug_csv,
        "--out_csv", args.out_csv,
        "--expert_weight", str(args.expert_weight),
        "--aug_weight", str(args.aug_weight),
        "--fill_nan_obs", str(args.fill_nan_obs),
        "--max_t", str(max_t),
    ]
    run(cmd)


if __name__ == "__main__":
    main()
