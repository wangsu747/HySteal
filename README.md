

# HySteal: Stealing Attacks on Multi-agent Deep Reinforcement

This is the source code for our paper. 
Hysteal is a Learning-based Coordinated Cyber-Physical Systems
HySteal is a hybrid imitation learning system that frames multi-agent model stealing combining behavioral cloning and adversarial imitation to recover high-fidelity MADRL policies in complex CPS settings.



HySteal 
- Train: runs the full pipeline (BC, dynamics, rollout, merge, HySteal training).
  ENV_TYPE=simple_tag_v3 N_ADVERSARIES=4 bash HySteal/run_all_single_scale.sh
- Evaluate: evaluates a model checkpoint on the selected environment.
  HySteal/evaluate.sh simple_tag_v3 /path/of/model.pt

MARL
- Train: trains MADDPG for the selected environment.
  ENV_TYPE=simple_tag_v3 bash MARL/train_maddpg.sh
- Evaluate: evaluates MADDPG for the selected environment.
  ENV_TYPE=simple_tag_v3 bash MARL/eval_maddpg.sh
