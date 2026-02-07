import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.il_csv_adapter import normalize_il_dataframe

class ExpertCSVDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 agent_order: list,
                 obs_dim: int,
                 action_n: int,
                 require_full: bool = True,
                 dtype=np.float32):
        super().__init__()
        self.csv_path = csv_path
        self.agent_order = list(agent_order)
        self.agent_to_idx = {a: i for i, a in enumerate(self.agent_order)}
        self.n_agents = len(self.agent_order)
        self.obs_dim = int(obs_dim)
        self.action_n = int(action_n)
        self.require_full = bool(require_full)

        df = pd.read_csv(csv_path, low_memory=False)
        df, _ = normalize_il_dataframe(df, obs_dim=self.obs_dim, fill_value=0.0, ensure_next_obs=True)

        need = ["episode", "t", "agent", "action_id"]
        for k in need:
            if k not in df.columns:
                raise ValueError(f"CSV missing required column: {k}")
        for i in range(self.obs_dim):
            if f"obs_{i}" not in df.columns or f"next_obs_{i}" not in df.columns:
                raise ValueError(f"CSV missing obs_{i} or next_obs_{i} after normalization")

        keep_cols = ["episode", "t", "agent", "action_id"] + \
                    [f"obs_{i}" for i in range(self.obs_dim)] + \
                    [f"next_obs_{i}" for i in range(self.obs_dim)]
        df = df[keep_cols].copy()

        df = df[df["agent"].isin(self.agent_order)].copy()

        grouped = df.groupby(["episode", "t"], sort=True)
        samples = []

        for (ep, tt), g in grouped:
            if self.require_full and len(g) != self.n_agents:
                continue

            if g["agent"].nunique() != len(g):
                continue

            try:
                g = g.sort_values(by="agent", key=lambda s: s.map(self.agent_to_idx))
            except Exception:
                continue

            if self.require_full:
                if list(g["agent"].values) != self.agent_order:
                    continue

            obs = g[[f"obs_{i}" for i in range(self.obs_dim)]].to_numpy(dtype=dtype)
            nobs = g[[f"next_obs_{i}" for i in range(self.obs_dim)]].to_numpy(dtype=dtype)

            if self.obs_dim >= 16 and "agent_0" in self.agent_to_idx:
                a0 = self.agent_to_idx["agent_0"]

                obs[a0, 14] = obs[a0, 0]
                obs[a0, 15] = obs[a0, 1]

                nobs[a0, 14] = nobs[a0, 0]
                nobs[a0, 15] = nobs[a0, 1]

            act_id = g["action_id"].to_numpy(dtype=np.int64)

            act_oh = np.zeros((self.n_agents, self.action_n), dtype=dtype)
            valid = (act_id >= 0) & (act_id < self.action_n)
            act_oh[np.arange(self.n_agents)[valid], act_id[valid]] = 1.0

            samples.append((
                obs.reshape(-1),
                act_oh.reshape(-1),
                nobs.reshape(-1),
            ))

        if len(samples) == 0:
            raise RuntimeError("No valid joint timesteps found. Check agent_order / require_full / CSV content.")

        self.states = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32)
        self.actions = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32)
        self.next_states = torch.tensor(np.stack([s[2] for s in samples]), dtype=torch.float32)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]
