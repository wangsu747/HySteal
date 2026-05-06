import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from SWARM.common import build_reviewer_ckpts, ensure_dir, python_exe, repo_root, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("SWARM aug")
    p.add_argument("--env", choices=["simple_tag_v3", "simple_spread_v3", "hvac"], required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--review_dir", default="")
    p.add_argument("--expert_ckpts", default="")
    p.add_argument("--num_reviewers", type=int, default=5)
    p.add_argument("--dynamics_ckpt", required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--keep_steps", type=int, required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--meta_json", default="")
    p.add_argument("--min_vote", type=float, default=0.5)
    p.add_argument("--min_avg_logp", type=float, default=-1.8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_attempts", type=int, default=200000)
    p.add_argument("--init_csv_path", default="")
    p.add_argument("--reset_from_t0_only", action="store_true")
    p.add_argument("--reset_noise_sigma", type=float, default=0.0)
    p.add_argument("--config", default="")
    p.add_argument("--start_csv", default="")
    p.add_argument("--start_mode", default="any")
    p.add_argument("--start_noise_sigma", type=float, default=0.01)
    p.add_argument("--rollout_noise_sigma", type=float, default=0.005)
    p.add_argument("--noise_dims", default="0-5,7-8,16-18")
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    expert_ckpts = args.expert_ckpts or build_reviewer_ckpts(args.review_dir, args.num_reviewers)
    if args.env == "hvac":
        cmd = [
            python_exe(),
            os.path.join(repo_root(), "SWARM", "hvac", "aug_hvac.py"),
            "--config", args.config or os.path.join(repo_root(), "SWARM", "configs", "campus_4agent.yml"),
            "--start_csv", args.start_csv,
            "--start_mode", args.start_mode,
            "--dynamics_ckpt", args.dynamics_ckpt,
            "--student_ckpt", args.student_ckpt,
            "--expert_ckpts", expert_ckpts,
            "--out_csv", args.out_csv,
            "--episodes", str(args.episodes),
            "--keep_steps", str(args.keep_steps),
            "--min_vote", str(args.min_vote),
            "--min_avg_logp", str(args.min_avg_logp),
            "--device", args.device,
            "--max_attempts", str(args.max_attempts),
            "--start_noise_sigma", str(args.start_noise_sigma),
            "--rollout_noise_sigma", str(args.rollout_noise_sigma),
            "--noise_dims", args.noise_dims,
        ]
        if args.meta_json:
            cmd.extend(["--save_meta_json", args.meta_json])
    else:
        cmd = [
            python_exe(),
            os.path.join(repo_root(), "SWARM", "aug_learned.py"),
            "--episodes", str(args.episodes),
            "--keep_steps", str(args.keep_steps),
            "--student_ckpt", args.student_ckpt,
            "--expert_ckpts", expert_ckpts,
            "--dynamics_ckpt", args.dynamics_ckpt,
            "--init_csv_path", args.init_csv_path,
            "--min_vote", str(args.min_vote),
            "--min_avg_logp", str(args.min_avg_logp),
            "--out_csv", args.out_csv,
            "--device", args.device,
            "--max_attempts", str(args.max_attempts),
        ]
        if args.meta_json:
            cmd.extend(["--save_meta_json", args.meta_json])
        if args.reset_from_t0_only:
            cmd.append("--reset_from_t0_only")
        cmd.extend(["--reset_noise_sigma", str(args.reset_noise_sigma)])
    run(cmd)


if __name__ == "__main__":
    main()
