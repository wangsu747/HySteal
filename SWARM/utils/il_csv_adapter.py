import ast
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def infer_obs_cols(df: pd.DataFrame, prefix: str = "obs_") -> List[str]:
    pairs = []
    for c in df.columns:
        if not c.startswith(prefix):
            continue
        tail = c[len(prefix):]
        if tail.isdigit():
            pairs.append((int(tail), c))
    pairs.sort(key=lambda x: x[0])
    return [c for _, c in pairs]


def _parse_vec(x) -> List[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except Exception:
            obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [float(v) for v in obj]
    return []


def ensure_action_id(df: pd.DataFrame) -> pd.DataFrame:
    if "action_id" in df.columns:
        df["action_id"] = pd.to_numeric(df["action_id"], errors="coerce").fillna(0).astype(int)
        return df
    if "action" in df.columns:
        df["action_id"] = pd.to_numeric(df["action"], errors="coerce").fillna(0).astype(int)
        return df
    raise ValueError("CSV needs `action_id` or `action`.")


def _ensure_obs_from_json(
    df: pd.DataFrame,
    json_col: str,
    out_prefix: str,
    obs_dim: Optional[int],
    fill_value: float,
) -> Tuple[pd.DataFrame, int]:
    cur_cols = infer_obs_cols(df, prefix=out_prefix)
    cur_dim = len(cur_cols)
    if obs_dim is None:
        obs_dim_target = cur_dim
    else:
        obs_dim_target = int(obs_dim)

    if json_col in df.columns:
        parsed = [_parse_vec(v) for v in df[json_col].tolist()]
        if obs_dim is None:
            max_len = max([len(v) for v in parsed] + [cur_dim, 0])
            obs_dim_target = max_len
        if obs_dim_target > cur_dim:
            for i in range(cur_dim, obs_dim_target):
                df[f"{out_prefix}{i}"] = np.nan
        for i in range(obs_dim_target):
            col = f"{out_prefix}{i}"
            vals = []
            for vec in parsed:
                if i < len(vec):
                    vals.append(float(vec[i]))
                else:
                    vals.append(np.nan)
            if col in df.columns:
                base = pd.to_numeric(df[col], errors="coerce")
                fill = pd.Series(vals, index=df.index, dtype="float64")
                df[col] = base.where(base.notna(), fill)
            else:
                df[col] = vals
    else:
        if obs_dim_target == 0:
            obs_dim_target = cur_dim

    final_cols = [f"{out_prefix}{i}" for i in range(obs_dim_target)]
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if final_cols:
        df[final_cols] = df[final_cols].fillna(float(fill_value)).astype(np.float32)
    return df, obs_dim_target


def normalize_il_dataframe(
    df: pd.DataFrame,
    obs_dim: Optional[int] = None,
    fill_value: float = 0.0,
    ensure_next_obs: bool = True,
) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    df = ensure_action_id(df)
    df, obs_dim_used = _ensure_obs_from_json(df, "obs_json", "obs_", obs_dim=obs_dim, fill_value=fill_value)
    if ensure_next_obs:
        df, _ = _ensure_obs_from_json(
            df, "next_obs_json", "next_obs_", obs_dim=obs_dim_used, fill_value=fill_value
        )
    return df, obs_dim_used
