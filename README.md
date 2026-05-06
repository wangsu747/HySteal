# SWARM

Multi-agent Deep Reinforcement Learning (MADRL) models are foundational to modern Cyber-Physical Systems (CPS) and edge computing. As these proprietary models become valuable intellectual property, recent campaigns highlight their vulnerability to model stealing. However, the feasibility of extracting MADRL systems remains largely unexplored due to challenges like high complexity and partial observability. To expose this vulnerability, we propose SWARM (Stealing via Weakly-supervised Action Representation Manifolds), a novel adversarial extraction framework. Rather than cloning a rigid deterministic policy, SWARM learns a continuous behavioral manifold capturing the target system's inherent uncertainty. It leverages an environment predictor to reconstruct hidden dynamics and a weak expert trajectory augmentation mechanism to mitigate observation bias from sparse data. A Multi-Agent Generative Adversarial Imitation Learning (MAGAIL) module uses these augmented trajectories to map the broader behavioral manifold, reconstructing latent cooperative and competitive dynamics. Evaluations across four CPS environments show SWARM achieves rapid convergence and high-fidelity extraction, scaling seamlessly with negligible overhead. Furthermore, SWARM's manifold-learning approach inherently compromises state-of-the-art ownership defenses by stripping behavioral watermarks, neutralizing backdoors, and bypassing policy obfuscation. Finally, we explore countermeasures via data poisoning and provide initial mitigation results. As one of the first frameworks to demonstrate the reality of MADRL extraction, SWARM exposes a potent attack vector while establishing a foundation for next-generation resilient CPS architectures.

Pipeline:

![Pipeline](picture/System-pipeline.png)


1. train weak experts from a small true-expert collected CSV
2. learn an environment from the same CSV
3. generate vote-filtered augmentation data
4. merge true expert data and generated data
5. re-learn dynamics from merged data
6. train the final policy
7. evaluate in the real environment

Examples below show only the required or most important arguments. Other options use script defaults.

## ThreatModel
![Pipeline](picture/threatmodel.png)

## Create the environment


```bash
conda create -n swarm python=3.9 -y
conda activate swarm
pip install -r requirements.txt
pip install pandas PyYAML click
```

Run commands from the repository root after activating the environment.

## Scripts

- `SWARM/weak.py`
- `SWARM/dyn.py`
- `SWARM/aug.py`
- `SWARM/merge.py`
- `SWARM/train.py`
- `SWARM/eval.py`

## Environment settings

simple_tag_v3

simple_spread_v3

hvac




## Step 1: weak experts

```bash
python3 SWARM/weak.py \
  --env <env> \
  --csv_path data/<env>/expert_small.csv \
  --out_dir runs/<env>/weak_bc
```

This creates `bc_best_seed0.pth` to `bc_best_seed5.pth`.

- `seed0`: student
- `seed1..seed5`: reviewers

## Step 2: training environment

```bash
python3 SWARM/dyn.py \
  --env <env> \
  --csv_path data/<env>/expert_small.csv \
  --out_path runs/<env>/dyn.pt
```

## Step 3: augmentation

### simple_tag_v3 / simple_spread_v3/hvac

```bash
python3 SWARM/aug.py \
  --env <env> \
  --student_ckpt runs/<env>/weak_bc/bc_best_seed0.pth \
  --review_dir runs/<env>/weak_bc \
  --dynamics_ckpt runs/<env>/dyn.pt \
  --init_csv_path data/<env>/expert_small.csv \
  --episodes <aug_episodes> \
  --keep_steps <keep_steps> \
  --out_csv runs/<env>/aug.csv
```

## Step 4: merge

### simple_spread_v3 / simple_tag_v3 / hvac

```bash
python3 SWARM/merge.py \
  --env <env> \
  --expert_csv data/<env>/expert_small.csv \
  --aug_csv runs/<env>/aug.csv \
  --out_csv runs/<env>/merged.csv
```

## Step 5: environemnt on merged data

```bash
python3 SWARM/dyn.py \
  --env <env> \
  --csv_path runs/<env>/merged.csv \
  --out_path runs/<env>/dyn_merged.pt
```

## Step 6: SWARM training

### simple_tag_v3 / simple_spread_v3

```bash
python3 SWARM/train.py \
  --env <env> \
  --csv_path runs/<env>/merged.csv \
  --dynamics_ckpt runs/<env>/dyn_merged.pt \
  --config SWARM/configs/config_multi.yml \
  --out_dir runs/<env>/final
```

### HVAC

```bash
python3 SWARM/train.py \
  --env hvac \
  --csv_path runs/hvac/merged.csv \
  --config SWARM/configs/campus_4agent.yml \
  --expert_policy_ckpt HVAC/checkpoints/shared_mappo_best.pt \
  --out_dir runs/hvac/final \
  --horizon 96 \
  --eval_episodes 100
```

## Step 7: evaluation

### simple_tag_v3 / simple_spread_v3 / hvac

```bash
python3 SWARM/eval.py \
  --env <env> \
  --train_out_dir runs/<env>/final \
  --out_json runs/<env>/final_eval.json \
  --episodes 100
```

### HVAC

Read:

```bash
runs/hvac/final/summary.json
```

## Train MARL

- Train: trains multi-agent reinforcement learning for the selected environment.
```
ENV_TYP bash MARL/train_maddpg.sh
```
- Evaluate: evaluates multi-agent reinforcement learning policy for the selected environment.
```
ENV_TYPE bash MARL/eval_maddpg.sh
```


## Notes

- `simple_tag_v3` and `simple_spread_v3` use learned-environment augmentation.
- `hvac` uses the HVAC rollout and reward-sweep path.
- `SWARM/train.py` includes evaluation for HVAC.
