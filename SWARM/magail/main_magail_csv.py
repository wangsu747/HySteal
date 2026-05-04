import os
import time
import sys
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_util import config_loader
from utils.il_csv_adapter import normalize_il_dataframe, infer_obs_cols
from hysteal.HySteal import HySteal

from bc_pretrain_from_csv import bc_pretrain

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

@click.command()
@click.option("--train_mode", type=bool, default=True, help="Train / Validate (True=train, False=eval-only)")
@click.option("--config_train", type=str, required=True, help="train yaml path")
@click.option("--config_eval", type=str, default="./config/config_validation.yml", help="eval yaml path (optional)")
@click.option("--num_agent", type=int, default=4, help="number of agents (must match CSV & config.general.agent_order)")
@click.option("--trial", type=int, default=81300, help="an id to separate logs/models")
@click.option("--eval_model_epoch", type=int, default=2, help="run evaluation every K epochs")
@click.option("--save_model_epoch", type=int, default=50, help="save checkpoint every K epochs")
@click.option("--save_model_path", type=str, required=True, help="directory to save models")
@click.option("--log_root", type=str, required=True, help="tensorboard log root directory")
@click.option("--plot", is_flag=True, help="plot losses at end")
@click.option("--bc_init", type=str, default=None, help="(optional) path to BC ckpt, ignored if HySteal doesn't use it")
@click.option("--expert_csv", type=str, default="", help="override config.general.expert_csv_path")

@click.option("--bc_pretrain_epochs", type=int, default=600, help="BC pretrain epochs before adversarial training")
@click.option("--bc_batch_size", type=int, default=4096, help="BC batch size")
@click.option("--bc_lr", type=float, default=3e-4, help="BC learning rate")
@click.option("--bc_ckpt_path", type=str, default=None, help="where to save BC ckpt (default: save_model_path/bc_pretrained_policy.pt)")
@click.option("--skip_bc_pretrain", is_flag=True, help="skip BC pretrain stage")
@click.option("--log_usage", is_flag=True, help="log per-epoch time/CPU/GPU/RAM (adversarial phase only)")
@click.option("--usage_out", type=str, default="", help="usage CSV path (default: data/usage_n{num_agent}_trial{trial}.csv)")
@click.option("--usage_gpu_index", type=int, default=0, help="GPU index for usage query")
def main(train_mode, config_train, config_eval, num_agent, trial,
         eval_model_epoch, save_model_epoch, save_model_path, log_root, plot, bc_init,
         expert_csv, bc_pretrain_epochs, bc_batch_size, bc_lr, bc_ckpt_path, skip_bc_pretrain,
         log_usage, usage_out, usage_gpu_index):
    def _infer_layout_from_expert_csv(csv_path: str, n_agents_cli: int):
        df_raw = pd.read_csv(csv_path, low_memory=False)
        df_norm, _ = normalize_il_dataframe(df_raw, obs_dim=None, fill_value=0.0, ensure_next_obs=False)
        if "agent" not in df_norm.columns:
            raise ValueError(f"expert csv missing `agent`: {csv_path}")
        seen = []
        sset = set()
        for a in df_norm["agent"].tolist():
            if a not in sset:
                sset.add(a)
                seen.append(str(a))
        if n_agents_cli > 0:
            n_adv = max(0, int(n_agents_cli) - 1)
            want = [f"adversary_{i}" for i in range(n_adv)] + ["agent_0"]
            if all(a in sset for a in want):
                seen = want
            elif len(seen) >= n_agents_cli:
                seen = seen[:int(n_agents_cli)]
        if len(seen) == 0:
            raise ValueError(f"cannot infer agent_order from csv: {csv_path}")
        obs_cols = infer_obs_cols(df_norm)
        if len(obs_cols) == 0:
            raise ValueError(f"cannot infer obs_* from csv: {csv_path}")
        action_n = int(pd.to_numeric(df_norm["action_id"], errors="coerce").fillna(0).max()) + 1
        return seen, len(obs_cols), action_n

    def _apply_layout_to_config(cfg, agent_order, obs_dim, action_n):
        n_agents = len(agent_order)
        cfg["general"]["agent_order"] = list(agent_order)
        cfg["general"]["num_agent"] = int(n_agents)
        cfg["general"]["obs_dim_per_agent"] = int(obs_dim)
        cfg["general"]["action_n"] = int(action_n)

        cfg["jointpolicy"]["agent"]["num_states"] = int(obs_dim)
        cfg["jointpolicy"]["agent"]["num_states_2"] = int(obs_dim * n_agents)
        cfg["jointpolicy"]["agent"]["num_actions"] = int(action_n)
        cfg["jointpolicy"]["agent"]["num_discrete_actions"] = int(action_n)
        cfg["jointpolicy"]["agent"]["discrete_actions_sections"] = (int(action_n),)
        cfg["jointpolicy"]["agent"]["num_agent"] = int(n_agents)
        cfg["jointpolicy"]["env"]["num_states"] = int(obs_dim + action_n)
        cfg["jointpolicy"]["env"]["num_states_2"] = int((obs_dim + action_n) * n_agents)
        cfg["jointpolicy"]["env"]["num_actions"] = int(obs_dim)
        cfg["jointpolicy"]["env"]["num_agent"] = int(n_agents)
        cfg["value"]["num_states"] = int(obs_dim)
        cfg["discriminator"]["num_states"] = int(obs_dim)
        cfg["discriminator"]["num_actions"] = int(action_n)
        cfg["discriminator"]["num_agent"] = int(n_agents)

    def _find_generator_policy(mail):
        candidates = [
            "P",
            "policy",
            "generator",
            "pi",
            "actor",
            "policy_net",
            "G",
            "gen",
        ]
        for name in candidates:
            if hasattr(mail, name):
                obj = getattr(mail, name)
                if obj is None:
                    continue
                if hasattr(obj, "parameters"):
                    return obj, name

        visible = list(getattr(mail, "__dict__", {}).keys())
        raise AttributeError(
            "[BC] Cannot find generator policy inside HySteal instance.\n"
            f"Visible keys: {visible}\n"
            "Try adding correct attribute name into candidates list."
        )

    if train_mode:
        cfg_path = config_train
        exp_name = f"MILD2_Train_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}_trial{trial}"
    else:
        cfg_path = config_eval
        exp_name = f"MILD2_Eval_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}_trial{trial}"

    config = config_loader(path=cfg_path)
    if isinstance(expert_csv, str) and expert_csv.strip():
        config["general"]["expert_csv_path"] = expert_csv.strip()

    try:
        inferred_order, inferred_obs_dim, inferred_action_n = _infer_layout_from_expert_csv(
            config["general"]["expert_csv_path"], int(num_agent)
        )
        _apply_layout_to_config(config, inferred_order, inferred_obs_dim, inferred_action_n)
    except Exception as e:
        print(f"[warn] auto-infer from expert csv failed, fallback to yaml config. reason: {e}")
        config["general"]["num_agent"] = int(num_agent)

    must_general = [
        "expert_csv_path", "agent_order", "obs_dim_per_agent", "action_n",
        "training_epochs", "expert_batch_size", "seed", "num_agent"
    ]
    missing = [k for k in must_general if k not in config.get("general", {})]
    if missing:
        raise KeyError(
            f"config.general missing keys: {missing}\n"
            f"Need at least: {must_general}\n"
            f"Current general keys: {list(config.get('general', {}).keys())}"
        )

    training_epochs = int(config["general"]["training_epochs"])

    config["general"]["num_agent"] = int(num_agent)

    print("=" * 80)
    print(f"Using config: {cfg_path}")
    print(f"exp_name: {exp_name}")
    print(f"expert_csv_path: {config['general']['expert_csv_path']}")
    print(f"agent_order: {config['general']['agent_order']}")
    print(f"obs_dim_per_agent: {config['general']['obs_dim_per_agent']}")
    print(f"action_n: {config['general']['action_n']}")
    print(f"training_epochs: {training_epochs}")
    print(f"num_agent (CLI override): {num_agent}")
    print("=" * 80)

    log_dir = os.path.join(log_root, f"trial_{trial}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)

    mail = HySteal(
        config=config,
        log_dir=log_dir,
        exp_name=exp_name,
        num_agent=num_agent,
        bc_init=bc_init
    )

    if not train_mode:
        mail.eval(epoch=0)
        print("[eval-only] done.")
        return

    if (not skip_bc_pretrain) and (bc_pretrain_epochs is not None) and (bc_pretrain_epochs > 0):
        expert_csv = config["general"]["expert_csv_path"]
        if bc_ckpt_path is None:
            bc_ckpt_path = os.path.join(save_model_path, "bc_pretrained_policy.pt")

        gen_policy, policy_name = _find_generator_policy(mail)
        print("=" * 80)
        print("[DEBUG] generator policy class:", type(gen_policy))
        print("[DEBUG] generator policy dict keys:", list(getattr(gen_policy, "__dict__", {}).keys()))
        cand = [n for n in dir(gen_policy) if any(k in n.lower() for k in
                                                  ["act", "action", "pi", "policy", "prob", "logp", "log_prob",
                                                   "evaluate", "dist", "forward", "sample"])]
        print("[DEBUG] callable-like names:", cand[:200])
        print("=" * 80)


        device = next(gen_policy.parameters()).device

        per_agent_dim = int(config["general"]["obs_dim_per_agent"])
        num_agent_cfg = int(config["general"]["num_agent"])
        joint_obs_dim = per_agent_dim
        agent_order = config["general"]["agent_order"]

        print("=" * 80)
        print(f"[BC] will pretrain generator policy: {policy_name}")
        print(f"[BC] expert_csv_path: {expert_csv}")
        print(f"[BC] per_agent_dim(csv): {per_agent_dim}")
        print(f"[BC] policy_obs_dim(joint): {joint_obs_dim}")
        print(f"[BC] epochs: {bc_pretrain_epochs}, batch_size: {bc_batch_size}, lr: {bc_lr}")
        print(f"[BC] ckpt save to: {bc_ckpt_path}")
        print("=" * 80)
        agent_order = config["general"]["agent_order"]

        saved_ckpt = bc_pretrain(
            policy=gen_policy,
            expert_csv_path=expert_csv,
            device=device,
            per_agent_obs_dim=int(config["general"]["obs_dim_per_agent"]),
            num_agent=int(config["general"]["num_agent"]),
            action_n=int(config["general"]["action_n"]),
            agent_order=config["general"]["agent_order"],
            epochs=bc_pretrain_epochs,
            batch_size=bc_batch_size,
            lr=bc_lr,
            save_path=bc_ckpt_path,
        )

        ckpt = torch.load(saved_ckpt, map_location=device)
        gen_policy.load_state_dict(ckpt["policy_state_dict"], strict=True)
        print(f"[BC] loaded pretrained weights -> {saved_ckpt}")

    all_td = []
    all_p = []
    all_v = []

    usage_rows = []
    psutil = None
    proc = None
    if log_usage:
        try:
            import psutil as _psutil
            psutil = _psutil
            proc = psutil.Process()
            proc.cpu_percent(interval=None)
        except Exception as e:
            print(f"[usage] psutil not available: {e}. CPU/RAM usage will be None.")

    def _gpu_stats():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(usage_gpu_index))
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(util), float(mem.used) / (1024 ** 2), float(mem.total) / (1024 ** 2)
        except Exception:
            pass
        try:
            cmd = [
                "nvidia-smi",
                f"--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(int(usage_gpu_index)),
            ]
            out = subprocess.check_output(cmd, text=True).strip()
            if not out:
                return None, None, None
            parts = [p.strip() for p in out.split(",")]
            return float(parts[0]), float(parts[1]), float(parts[2])
        except Exception:
            return None, None, None

    for epoch in range(1, training_epochs + 1):
        epoch_t0 = time.time()
        td_loss, v_loss, p_loss = mail.train(epoch)
        epoch_t1 = time.time()

        all_td.append(float(td_loss))
        all_v.append(float(v_loss))
        all_p.append(float(p_loss))

        if log_usage:
            cpu_pct = None
            ram_gb = None
            ram_pct = None
            if proc is not None and psutil is not None:
                try:
                    cpu_pct = float(proc.cpu_percent(interval=None))
                    mem = proc.memory_info().rss / (1024 ** 3)
                    ram_gb = float(mem)
                    vm = psutil.virtual_memory()
                    ram_pct = float(vm.percent)
                except Exception:
                    pass
            gpu_util, gpu_mem_used, gpu_mem_total = _gpu_stats()
            usage_rows.append({
                "epoch": int(epoch),
                "epoch_time_sec": float(epoch_t1 - epoch_t0),
                "cpu_percent": cpu_pct,
                "ram_used_gb": ram_gb,
                "ram_percent": ram_pct,
                "gpu_util_percent": gpu_util,
                "gpu_mem_used_mb": gpu_mem_used,
                "gpu_mem_total_mb": gpu_mem_total,
            })

        if eval_model_epoch > 0 and (epoch % eval_model_epoch == 0):
            mail.eval(epoch)

        if save_model_epoch > 0 and (epoch % save_model_epoch == 0):
            mail.save_model(save_model_path)
            print(f"[saved] -> {save_model_path}")

    if plot:
        xs = np.arange(1, training_epochs + 1)

        plt.figure()
        plt.plot(xs, all_td, label="td_loss_mean")
        plt.legend()
        plt.grid(True)
        plt.title("Transformer Discriminator (BCE) Loss")
        plt.savefig(os.path.join(save_model_path, f"{exp_name}_td_loss.png"), dpi=150)

        plt.figure()
        plt.plot(xs, all_p, label="p_loss_mean")
        plt.plot(xs, all_v, label="v_loss_mean")
        plt.legend()
        plt.grid(True)
        plt.title("PPO Loss")
        plt.savefig(os.path.join(save_model_path, f"{exp_name}_ppo_loss.png"), dpi=150)

        print("[plot] saved plots to:", save_model_path)

    if log_usage:
        if not usage_out:
            os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
            usage_out = os.path.join(PROJECT_ROOT, "data", f"usage_n{int(num_agent)}_trial{int(trial)}.csv")
        df_usage = pd.DataFrame(usage_rows)
        df_usage.to_csv(usage_out, index=False)
        print(f"[usage] saved -> {usage_out}")


if __name__ == "__main__":
    main()
