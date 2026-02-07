import time
import numpy as np
import gfootball.env as fe


class GRFParallelEnv:

    BALL_X = 0
    BALL_Y = 1

    def __init__(
        self,
        env_name: str,
        n_left: int,
        n_right: int = 0,
        representation: str = "simple115v2",
        rewards: str = "scoring,checkpoints",
        render: bool = False,
        episode_length: int = 200,

        shaping: bool = True,
        w_progress_team: float = 0.02,
        w_progress_owner: float = 0.03,
        w_back_penalty: float = 0.04,
        w_possession: float = 0.00,
        w_pass_success: float = 0.015,
        w_shot: float = 0.5,
        w_goal_bonus: float = 5.0,

        goal_reward_threshold: float = 0.95,
        goal_cooldown_steps: int = 8,

        debug: bool = False,
        debug_actions: bool = False,
        debug_first_n_steps: int = 30,
        force_action = None,

        step_sleep: float = 0.0,
        goal_sleep: float = 0.0,
    ):
        self.n_left = int(n_left)
        self.n_right = int(n_right)
        self.n_agents = self.n_left + self.n_right
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        self.shaping = bool(shaping)
        self.w_progress_team = float(w_progress_team)
        self.w_progress_owner = float(w_progress_owner)
        self.w_back_penalty = float(w_back_penalty)
        self.w_possession = float(w_possession)
        self.w_pass_success = float(w_pass_success)
        self.w_shot = float(w_shot)
        self.w_goal_bonus = float(w_goal_bonus)

        self.goal_reward_threshold = float(goal_reward_threshold)
        self.goal_cooldown_steps = int(goal_cooldown_steps)

        self.debug = bool(debug)
        self.debug_actions = bool(debug_actions)
        self.debug_first_n_steps = int(debug_first_n_steps)
        self.force_action = force_action if force_action is None else int(force_action)

        self.step_sleep = float(step_sleep)
        self.goal_sleep = float(goal_sleep)

        self.env = fe.create_environment(
            env_name=env_name,
            representation=representation,
            number_of_left_players_agent_controls=self.n_left,
            number_of_right_players_agent_controls=self.n_right,
            rewards=rewards,
            render=render,
        )

        self.max_episode_steps = int(episode_length)
        self.cur_step = 0

        self.action_space = self.env.action_space

        if hasattr(self.action_space, "nvec"):
            self._action_mode = "multidiscrete"
            self.act_dim = int(np.asarray(self.action_space.nvec)[0])
        elif hasattr(self.action_space, "n"):
            self._action_mode = "discrete"
            self.act_dim = int(self.action_space.n)
        else:
            raise ValueError(f"Unsupported action_space type: {type(self.action_space)}")

        raw_obs = self.env.reset()
        obs2 = self._format_obs(raw_obs)
        self.obs_dim = int(obs2.shape[-1])

        self._reset_stats()

        self.last_ball_x = None
        self.last_ball_owned_team = None
        self.last_ball_owned_player = None

        self.last_score_left = 0
        self.last_score_right = 0

        self._last_goal_step = -10**9

        if self.debug:
            print("[DEBUG] action_space:", type(self.action_space), "mode:", self._action_mode, "act_dim:", self.act_dim)
            print("[DEBUG] obs_dim:", self.obs_dim, "n_agents:", self.n_agents)

    def _format_obs(self, obs):

        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != self.n_agents:
            if arr.shape[0] == 1:
                arr = np.repeat(arr, self.n_agents, axis=0)
            else:
                raise ValueError(f"Obs first dim {arr.shape[0]} != n_agents {self.n_agents}")
        return arr

    def _format_reward(self, reward):

        if np.isscalar(reward):
            return np.full((self.n_agents,), float(reward), dtype=np.float32)
        arr = np.asarray(reward, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            return np.full((self.n_agents,), float(arr.item()), dtype=np.float32)
        if arr.size != self.n_agents:
            raise ValueError(f"Reward size {arr.size} != n_agents {self.n_agents}")
        return arr.reshape(self.n_agents)

    def _format_done(self, done):

        if isinstance(done, (bool, np.bool_)):
            return np.full((self.n_agents,), bool(done))
        arr = np.asarray(done).reshape(-1)
        if arr.size == 1:
            return np.full((self.n_agents,), bool(arr.item()))
        if arr.size != self.n_agents:
            raise ValueError(f"Done size {arr.size} != n_agents {self.n_agents}")
        return arr.astype(bool).reshape(self.n_agents)

    def _extract_info_scalar(self, info):

        if isinstance(info, dict):
            return info
        if isinstance(info, (list, tuple)) and len(info) > 0:
            prefer_keys = ["score", "left_score", "right_score", "ball_owned_team", "ball_owned_player", "events"]
            for k in prefer_keys:
                for d in info:
                    if isinstance(d, dict) and (k in d):
                        return d
            for d in info:
                if isinstance(d, dict):
                    return d
        return {}

    def _get_ball_x(self, obs_row):
        return float(obs_row[self.BALL_X])

    def _get_score(self, info_dict):
        score = info_dict.get("score", None)
        if isinstance(score, (list, tuple, np.ndarray)) and len(score) == 2:
            try:
                return int(score[0]), int(score[1])
            except Exception:
                pass
        if "left_score" in info_dict and "right_score" in info_dict:
            try:
                return int(info_dict["left_score"]), int(info_dict["right_score"])
            except Exception:
                pass
        return self.last_score_left, self.last_score_right

    def _get_ball_owned(self, info_dict):
        bot = info_dict.get("ball_owned_team", None)
        bop = info_dict.get("ball_owned_player", None)
        try:
            bot = None if bot is None else int(bot)
            bop = None if bop is None else int(bop)
        except Exception:
            return None, None
        return bot, bop

    def _reset_stats(self):
        self.stats = {
            "steps": 0,
            "shots": 0,
            "passes": 0,
            "goals_for": 0,
            "goals_against": 0,
            "possession_steps": 0,
            "ball_progress": 0.0,
        }

    def _goal_from_info_or_reward(self, info0, base_r):

        if (self.cur_step - self._last_goal_step) < self.goal_cooldown_steps:
            return False, False

        old_sl, old_sr = self.last_score_left, self.last_score_right
        sl, sr = self._get_score(info0)
        if (sl != old_sl) or (sr != old_sr):
            self.last_score_left, self.last_score_right = sl, sr
            self._last_goal_step = self.cur_step
            return (sl > old_sl), (sr > old_sr)

        mx = float(np.max(base_r)) if base_r.size else 0.0
        mn = float(np.min(base_r)) if base_r.size else 0.0
        if mx >= self.goal_reward_threshold:
            self._last_goal_step = self.cur_step
            return True, False
        if mn <= -self.goal_reward_threshold:
            self._last_goal_step = self.cur_step
            return False, True
        return False, False

    def reset(self):
        raw_obs = self.env.reset()
        obs = self._format_obs(raw_obs)

        self.cur_step = 0
        self._reset_stats()

        self.last_ball_x = self._get_ball_x(obs[0])
        self.last_ball_owned_team = None
        self.last_ball_owned_player = None

        self.last_score_left = 0
        self.last_score_right = 0
        self._last_goal_step = -10**9

        return {f"agent_{i}": obs[i].copy() for i in range(self.n_agents)}

    def step(self, action_dict):
        if self._action_mode == "discrete":
            a0 = int(action_dict["agent_0"])
            if self.force_action is not None:
                a0 = int(self.force_action)
            next_obs, reward, done, info = self.env.step(a0)
            exec_actions = [a0]
        else:
            actions = np.array([int(action_dict[f"agent_{i}"]) for i in range(self.n_agents)], dtype=np.int32)
            if self.force_action is not None:
                actions[:] = int(self.force_action)
            next_obs, reward, done, info = self.env.step(actions)
            exec_actions = actions.tolist()

        info0 = self._extract_info_scalar(info)

        self.cur_step += 1
        self.stats["steps"] += 1

        next_obs2 = self._format_obs(next_obs)
        base_r = self._format_reward(reward)
        done_arr = self._format_done(done)

        left_scored, right_scored = self._goal_from_info_or_reward(info0, base_r)
        if left_scored:
            self.stats["goals_for"] += 1
        if right_scored:
            self.stats["goals_against"] += 1

        ball_x = self._get_ball_x(next_obs2[0])
        if self.last_ball_x is None:
            self.last_ball_x = ball_x
        delta_x = ball_x - self.last_ball_x
        self.last_ball_x = ball_x
        self.stats["ball_progress"] += float(delta_x)

        bot, bop = self._get_ball_owned(info0)
        owner_agent = None
        if bot == 0 and bop is not None and 0 <= bop < self.n_left:
            owner_agent = int(bop)

        pass_success = False
        prev_owner_agent = None
        if self.last_ball_owned_team == 0 and self.last_ball_owned_player is not None:
            if 0 <= self.last_ball_owned_player < self.n_left:
                prev_owner_agent = int(self.last_ball_owned_player)

        if self.last_ball_owned_team == 0 and bot == 0:
            if (bop is not None) and (self.last_ball_owned_player is not None) and (bop != self.last_ball_owned_player):
                pass_success = True

        self.last_ball_owned_team, self.last_ball_owned_player = bot, bop

        shot_happened = any(int(a) == 12 for a in exec_actions)
        if shot_happened:
            self.stats["shots"] += 1
        if pass_success:
            self.stats["passes"] += 1
        if bot == 0:
            self.stats["possession_steps"] += 1

        shaped = np.zeros((self.n_agents,), dtype=np.float32)
        if self.shaping:
            shaped += float(self.w_progress_team * delta_x)

            if delta_x < 0:
                shaped += float(self.w_back_penalty * delta_x)

            if owner_agent is not None:
                shaped[owner_agent] += float(self.w_progress_owner * delta_x)
                shaped[owner_agent] += float(self.w_possession)

                if shot_happened:
                    danger = max(0.0, min(1.0, (ball_x + 1.0) / 2.0))
                    if danger > 0.8:
                        shaped[owner_agent] += float(self.w_shot * (danger - 0.8) / 0.2)
                    else:
                        shaped[owner_agent] -= 0.02

            if pass_success and (prev_owner_agent is not None) and (owner_agent is not None):
                shaped[prev_owner_agent] += float(0.01)
                shaped[owner_agent] += float(self.w_pass_success)

            if left_scored:
                shaped += float(self.w_goal_bonus)
            if right_scored:
                shaped -= float(self.w_goal_bonus)

        total_r = base_r + shaped

        time_limit = False
        if self.cur_step >= self.max_episode_steps:
            done_arr[:] = True
            time_limit = True

        next_obs_dict = {f"agent_{i}": next_obs2[i].copy() for i in range(self.n_agents)}
        reward_dict = {f"agent_{i}": float(total_r[i]) for i in range(self.n_agents)}
        done_dict = {f"agent_{i}": bool(done_arr[i]) for i in range(self.n_agents)}

        info_pack = dict(info0) if isinstance(info0, dict) else {}
        info_pack["maddpg_stats"] = dict(self.stats)
        info_pack["time_limit"] = bool(time_limit)
        info_dict = {f"agent_{i}": info_pack for i in range(self.n_agents)}

        if self.debug_actions and self.cur_step <= self.debug_first_n_steps:
            print(
                f"[DBG step {self.cur_step:03d}] act={exec_actions} "
                f"ball_x={ball_x:+.3f} dx={delta_x:+.3f} "
                f"r(min/max)={float(np.min(base_r)):+.3f}/{float(np.max(base_r)):+.3f} "
                f"goals_for={self.stats['goals_for']} goals_against={self.stats['goals_against']}"
            )

        if self.step_sleep > 0:
            time.sleep(self.step_sleep)
        if (left_scored or right_scored) and self.goal_sleep > 0:
            time.sleep(self.goal_sleep)

        return next_obs_dict, reward_dict, done_dict, info_dict

    def sample_action_dict(self):
        a = self.action_space.sample()
        if isinstance(a, (list, tuple, np.ndarray)):
            return {f"agent_{i}": int(a[i]) for i in range(self.n_agents)}
        else:
            return {f"agent_{i}": int(a) for i in range(self.n_agents)}

    def close(self):
        self.env.close()
