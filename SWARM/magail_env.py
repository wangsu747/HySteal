#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from magail.MAGAIL import MAGAIL
from SWARM.lenv import make_learned_env_factory
from utils.torch_util import device


class MAGAILLearnedEnv(MAGAIL):
    def __init__(
        self,
        config,
        log_dir,
        exp_name,
        num_agent,
        dynamics_ckpt_path: str,
        init_csv_path: str,
        reset_from_t0_only: bool = True,
        reset_noise_sigma: float = 0.0,
        bc_init: str = None,
    ):
        self.dynamics_ckpt_path = dynamics_ckpt_path
        self.init_csv_path = init_csv_path
        self.reset_from_t0_only = bool(reset_from_t0_only)
        self.reset_noise_sigma = float(reset_noise_sigma)
        super().__init__(
            config=config,
            log_dir=log_dir,
            exp_name=exp_name,
            num_agent=num_agent,
            bc_init=bc_init,
        )

    def _build_make_env_fn(self):
        return make_learned_env_factory(
            dynamics_ckpt_path=self.dynamics_ckpt_path,
            init_csv_path=self.init_csv_path,
            max_cycles=int(self.config["jointpolicy"]["trajectory_length"]),
            reset_from_t0_only=self.reset_from_t0_only,
            reset_noise_sigma=self.reset_noise_sigma,
            device=str(device),
        )
