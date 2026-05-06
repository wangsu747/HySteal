import os
import subprocess
import sys
from typing import List


ENV_SPECS = {
    "simple_tag_v3": {
        "agent_order": ["adversary_0", "adversary_1", "adversary_2", "adversary_3", "agent_0"],
        "num_agent": 5,
        "max_cycles": 20,
    },
    "simple_spread_v3": {
        "agent_order": ["agent_0", "agent_1", "agent_2"],
        "num_agent": 3,
        "max_cycles": 25,
    },
    "hvac": {
        "agent_order": ["office", "hospital", "school", "retail"],
        "num_agent": 4,
        "max_cycles": 96,
    },
}


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def python_exe() -> str:
    return sys.executable


def agent_order_csv(env_name: str, override: str = "") -> str:
    if override.strip():
        return override.strip()
    return ",".join(ENV_SPECS[env_name]["agent_order"])


def agent_order_list(env_name: str, override: str = "") -> List[str]:
    return [x.strip() for x in agent_order_csv(env_name, override).split(",") if x.strip()]


def num_agent(env_name: str) -> int:
    return int(ENV_SPECS[env_name]["num_agent"])


def max_cycles(env_name: str) -> int:
    return int(ENV_SPECS[env_name]["max_cycles"])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_reviewer_ckpts(review_dir: str, num_reviewers: int) -> str:
    paths = []
    for seed in range(1, num_reviewers + 1):
        paths.append(os.path.join(review_dir, f"bc_best_seed{seed}.pth"))
    return ",".join(paths)


def run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root(), check=True)

