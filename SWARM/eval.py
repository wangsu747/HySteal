import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.common import agent_order_csv, ensure_dir, max_cycles, num_agent, python_exe, repo_root, run


def _infer_jointpolicy_path(model_dir: str) -> str:
    files = [x for x in os.listdir(model_dir) if x.endswith("_JointPolicy.pt")]
    if not files:
        raise FileNotFoundError(f"no *_JointPolicy.pt under {model_dir}")
    files.sort()
    return os.path.join(model_dir, files[-1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SWARM eval")
    p.add_argument("--env", choices=["simple_tag_v3", "simple_spread_v3"], required=True)
    p.add_argument("--jointpolicy_path", default="")
    p.add_argument("--train_out_dir", default="")
    p.add_argument("--out_json", required=True)
    p.add_argument("--agent_order", default="")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_cycles", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(os.path.dirname(args.out_json) or ".")
    jointpolicy_path = args.jointpolicy_path or _infer_jointpolicy_path(os.path.join(args.train_out_dir, "models"))
    cmd = [
        python_exe(),
        os.path.join(repo_root(), "SWARM", "eval_env.py"),
        "--env_name", args.env,
        "--agent_order", agent_order_csv(args.env, args.agent_order),
        "--episodes", str(args.episodes),
        "--max_cycles", str(args.max_cycles or max_cycles(args.env)),
        "--seed", str(args.seed),
        "--device", args.device,
        "--magail_jointpolicy_path", jointpolicy_path,
        "--save_json", args.out_json,
    ]
    if args.env == "simple_tag_v3":
        cmd.extend(["--num_adversaries", "4", "--num_good", "1"])
    else:
        cmd.extend(["--num_agents", str(num_agent(args.env))])
    run(cmd)


if __name__ == "__main__":
    main()
