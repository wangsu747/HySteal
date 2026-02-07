Run

HySteal train
ENV_TYPE=simple_tag_v3 N_ADVERSARIES=4 bash HySteal/run_all_single_scale.sh

HySteal evaluate
HySteal/evaluate.sh simple_tag_v3 /path/of/model.pt

MARL train
ENV_TYPE=simple_tag_v3 bash MARL/train_maddpg.sh

MARL evaluate
ENV_TYPE=simple_tag_v3 bash MARL/eval_maddpg.sh
