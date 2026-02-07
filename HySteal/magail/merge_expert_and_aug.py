


import argparse
import os
import re
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.il_csv_adapter import normalize_il_dataframe

OBS_RE = re.compile(r"^obs_(\d+)$")
NOBS_RE = re.compile(r"^next_obs_(\d+)$")

def get_obs_cols(df):
    cols = []
    for c in df.columns:
        m = OBS_RE.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    return [c for _, c in sorted(cols, key=lambda x: x[0])]

def get_next_obs_cols(df):
    cols = []
    for c in df.columns:
        m = NOBS_RE.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    return [c for _, c in sorted(cols, key=lambda x: x[0])]

def ensure_cols(df, cols, fill_value=np.nan):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df

def coerce_int_col(df, col, default=0):
    if col not in df.columns:
        df[col] = default
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)
    return df

def force_agent0_speed_copy(df: pd.DataFrame, agent_name: str = "agent_0"):






    need_cols = ["agent", "obs_0", "obs_1", "obs_14", "obs_15",
                 "next_obs_0", "next_obs_1", "next_obs_14", "next_obs_15"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    m = (df["agent"] == agent_name)
    if m.any():
        df.loc[m, "obs_14"] = df.loc[m, "obs_0"]
        df.loc[m, "obs_15"] = df.loc[m, "obs_1"]
        df.loc[m, "next_obs_14"] = df.loc[m, "next_obs_0"]
        df.loc[m, "next_obs_15"] = df.loc[m, "next_obs_1"]
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert_csv", type=str, required=True)
    p.add_argument("--aug_csv", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--expert_weight", type=float, default=1.0)
    p.add_argument("--aug_weight", type=float, default=0.5)
    p.add_argument("--fill_nan_obs", type=float, default=0.0, help="fill NaNs in obs/next_obs with this value")
    p.add_argument("--max_t", type=int, default=20, help="keep timesteps t in [0, max_t-1]")
    args = p.parse_args()

    df_e = pd.read_csv(args.expert_csv, low_memory=False)
    df_a = pd.read_csv(args.aug_csv, low_memory=False)
    df_e, obs_dim_e = normalize_il_dataframe(df_e, obs_dim=None, fill_value=args.fill_nan_obs, ensure_next_obs=True)
    df_a, obs_dim_a = normalize_il_dataframe(df_a, obs_dim=None, fill_value=args.fill_nan_obs, ensure_next_obs=True)
    obs_dim = max(obs_dim_e, obs_dim_a)
    if obs_dim > 0:
        df_e, _ = normalize_il_dataframe(df_e, obs_dim=obs_dim, fill_value=args.fill_nan_obs, ensure_next_obs=True)
        df_a, _ = normalize_il_dataframe(df_a, obs_dim=obs_dim, fill_value=args.fill_nan_obs, ensure_next_obs=True)

    df_e = coerce_int_col(df_e, "t", default=0)
    df_a = coerce_int_col(df_a, "t", default=0)
    df_e = df_e[df_e["t"] < args.max_t].copy()
    df_a = df_a[df_a["t"] < args.max_t].copy()

    e_obs = set(get_obs_cols(df_e))
    a_obs = set(get_obs_cols(df_a))
    e_nobs = set(get_next_obs_cols(df_e))
    a_nobs = set(get_next_obs_cols(df_a))

    obs_cols = sorted(list(e_obs | a_obs), key=lambda c: int(c.split("_")[1]))
    next_obs_cols = sorted(list(e_nobs | a_nobs), key=lambda c: int(c.rsplit("_", 1)[1]))

    base_cols = ["episode", "t", "agent", "action_id", "reward", "done"]
    optional_cols = [
        "action_json", "obs_json", "next_obs_json",
        "vote_ratio", "avg_logp"
    ]

    df_e["source"] = "expert"
    df_a["source"] = "aug"
    df_e["weight"] = float(args.expert_weight)
    df_a["weight"] = float(args.aug_weight)

    for c in base_cols:
        if c not in df_e.columns:
            df_e[c] = np.nan
        if c not in df_a.columns:
            df_a[c] = np.nan

    all_cols = base_cols + optional_cols + obs_cols + next_obs_cols + ["source", "weight"]
    df_e = ensure_cols(df_e, all_cols, fill_value=np.nan)
    df_a = ensure_cols(df_a, all_cols, fill_value=np.nan)

    df_e = df_e[all_cols]
    df_a = df_a[all_cols]

    fill_cols = obs_cols + next_obs_cols
    nan_e = int(df_e[fill_cols].isna().sum().sum())
    nan_a = int(df_a[fill_cols].isna().sum().sum())
    if nan_e > 0 or nan_a > 0:
        print(f"[warn] NaNs in obs/next_obs: expert={nan_e}, aug={nan_a}. Filling with {args.fill_nan_obs}")
        df_e[fill_cols] = df_e[fill_cols].fillna(args.fill_nan_obs)
        df_a[fill_cols] = df_a[fill_cols].fillna(args.fill_nan_obs)

    df_out = pd.concat([df_e, df_a], ignore_index=True)

    df_out = force_agent0_speed_copy(df_out, agent_name="agent_0")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved merged CSV: {args.out_csv}")
    print(f"rows: expert={len(df_e)} aug={len(df_a)} merged={len(df_out)}")
    print(f"kept timesteps: t in [0, {args.max_t-1}]")
    print("columns:", len(df_out.columns))

if __name__ == "__main__":
    main()
