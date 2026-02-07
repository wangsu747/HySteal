Run

Train
ENV_TYPE=simple_tag_v3 N_ADVERSARIES=4 bash run_all_single_scale.sh

Evaluate
./evaluate.sh simple_tag_v3 /path/of/model.pt
