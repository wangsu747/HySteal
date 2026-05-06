from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


TARGET_COLUMNS = [
    "episode",
    "t",
    "agent",
    "obs_json",
    "action",
    "next_obs_json",
    "reward",
    "done",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Convert HVAC exported IL CSV into the older minimal MAGAIL/MADDPG-style transition format."
    )
    p.add_argument("--in_csv", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument(
        "--agent_map_json",
        type=str,
        default="",
        help='Optional JSON mapping like \'{"office":"adversary_0","hospital":"adversary_1"}\'.',
    )
    return p.parse_args()


def load_agent_map(raw: str) -> Dict[str, str]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("--agent_map_json must decode to a JSON object.")
    return {str(k): str(v) for k, v in obj.items()}


def main() -> None:
    args = parse_args()
    in_csv = Path(args.in_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agent_map = load_agent_map(args.agent_map_json)

    with in_csv.open("r", newline="", encoding="utf-8") as src_f, out_csv.open(
        "w", newline="", encoding="utf-8"
    ) as dst_f:
        reader = csv.DictReader(src_f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {in_csv}")

        missing = [c for c in ["episode", "t", "agent", "obs_json", "action", "next_obs_json", "reward", "done"] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns for legacy export: {missing}")

        writer = csv.DictWriter(dst_f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()

        row_count = 0
        for row in reader:
            agent = str(row["agent"])
            row_out = {
                "episode": row["episode"],
                "t": row["t"],
                "agent": agent_map.get(agent, agent),
                "obs_json": row["obs_json"],
                "action": row["action"],
                "next_obs_json": row["next_obs_json"],
                "reward": row["reward"],
                "done": row["done"],
            }
            writer.writerow(row_out)
            row_count += 1

    print(f"[done] wrote legacy IL CSV: {out_csv}")
    print(f"[info] rows: {row_count}")
    print(f"[info] columns: {TARGET_COLUMNS}")
    if agent_map:
        print(f"[info] applied agent map: {agent_map}")


if __name__ == "__main__":
    main()
