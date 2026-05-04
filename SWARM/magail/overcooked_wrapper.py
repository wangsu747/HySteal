import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        yield
class OvercookedParallelEnv:
    def __init__(
        self,
        layout_name="cramped_room",
        episode_length=400,
        shape_reward=False,
        render=False,
        debug_api=False,
        time_penalty=0.0,
        quiet=True,
    ):
        self.quiet = bool(quiet)
        self.render_enabled = bool(render)
        self.layout_name = str(layout_name)
        self.episode_length = int(episode_length)
        self.shape_reward = bool(shape_reward)
        self.debug_api = bool(debug_api)
        self.time_penalty = float(time_penalty)

        with suppress_output(self.quiet):
            self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
            self.env = OvercookedEnv.from_mdp(self.mdp, horizon=self.episode_length)

        self.n_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self._actions = list(Action.ALL_ACTIONS)
        self.act_dim = len(self._actions)

        self._reset_stats()

        obs = self.reset()
        self.obs_dim = int(obs["agent_0"].shape[-1])

        if self.debug_api:
            self._print_api()

    def _print_api(self):
        env_keys = [k for k in dir(self.env) if ("feat" in k) or ("ob" in k) or ("render" in k)]
        mdp_keys = [k for k in dir(self.mdp) if ("feat" in k) or ("enc" in k) or ("ob" in k)]
        print("[OvercookedWrapper] env keys:", env_keys)
        print("[OvercookedWrapper] mdp keys:", mdp_keys)

    def _reset_stats(self):
        self.stats = {
            "steps": 0,
            "sparse_reward_sum": 0.0,
            "deliveries": 0,
        }

    def _to_1d_float(self, x):
        arr = np.asarray(x, dtype=np.float32)
        return arr.reshape(-1)

    def _format_obs(self, feat):
        try:
            arr = np.asarray(feat, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                return {
                    "agent_0": arr[0].reshape(-1),
                    "agent_1": arr[1].reshape(-1),
                }
        except Exception:
            arr = None

        if isinstance(feat, (list, tuple)) and len(feat) >= 2:
            return {"agent_0": self._to_1d_float(feat[0]), "agent_1": self._to_1d_float(feat[1])}

        if isinstance(feat, dict):
            if 0 in feat and 1 in feat:
                return {"agent_0": self._to_1d_float(feat[0]), "agent_1": self._to_1d_float(feat[1])}
            for k0, k1 in [("player_0", "player_1"), ("p0", "p1"), ("agent_0", "agent_1")]:
                if k0 in feat and k1 in feat:
                    return {"agent_0": self._to_1d_float(feat[k0]), "agent_1": self._to_1d_float(feat[k1])}

            vals = [self._to_1d_float(v) for v in feat.values()]
            flat = self._to_1d_float(np.concatenate(vals, axis=0))
            return {"agent_0": flat.copy(), "agent_1": flat.copy()}

        flat = self._to_1d_float(feat)
        return {"agent_0": flat.copy(), "agent_1": flat.copy()}

    def _featurize(self, state):
        if hasattr(self.mdp, "lossless_state_encoding"):
            return self.mdp.lossless_state_encoding(state)

        if hasattr(self.env, "featurize_state"):
            return self.env.featurize_state(state)

        if hasattr(self.mdp, "featurize_state"):
            return self.mdp.featurize_state(state)

        if hasattr(self.mdp, "featurize_state_mdp"):
            return self.mdp.featurize_state_mdp(state)

        raise AttributeError("No featurize method found on mdp/env.")

    def _safe_render(self):
        if not self.render_enabled:
            return
        if hasattr(self.env, "render"):
            try:
                self.env.render()
                return
            except TypeError:
                pass
            except Exception:
                pass
            try:
                self.env.render(mode="human")
            except Exception:
                pass

    def reset(self):
        with suppress_output(self.quiet):
            self.env.reset()
        self._reset_stats()

        state = getattr(self.env, "state", None)
        if state is None:
            with suppress_output(self.quiet):
                state = self.env.reset()

        feat = self._featurize(state)
        return self._format_obs(feat)

    def sample_action_dict(self):
        return {
            "agent_0": int(np.random.randint(self.act_dim)),
            "agent_1": int(np.random.randint(self.act_dim)),
        }

    def step(self, action_dict):
        a0 = int(action_dict["agent_0"])
        a1 = int(action_dict["agent_1"])
        joint_action = (self._actions[a0], self._actions[a1])

        with suppress_output(self.quiet):
            next_state, sparse_r, done, info = self.env.step(joint_action)
        self._safe_render()

        team_r = float(sparse_r)

        shaped_r = 0.0
        if isinstance(info, dict):
            shaped_r = float(info.get("shaped_r", info.get("reward_shaping", 0.0)))

        if self.shape_reward:
            team_r += shaped_r
            if self.time_penalty != 0.0:
                team_r += float(self.time_penalty)

        feat = self._featurize(next_state)
        next_obs = self._format_obs(feat)

        self.stats["steps"] += 1
        self.stats["sparse_reward_sum"] += float(sparse_r)
        if float(sparse_r) > 0:
            self.stats["deliveries"] += 1

        reward = {"agent_0": team_r / 2.0, "agent_1": team_r / 2.0}
        done_dict = {"agent_0": bool(done), "agent_1": bool(done)}

        info_pack = {}
        if isinstance(info, dict):
            info_pack.update(info)
        info_pack["maddpg_stats"] = dict(self.stats)

        info_dict = {"agent_0": info_pack, "agent_1": info_pack}

        return next_obs, reward, done_dict, info_dict

    def close(self):
        if hasattr(self.env, "close"):
            try:
                self.env.close()
            except Exception:
                pass
