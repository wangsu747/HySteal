set -euo pipefail


ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

ENV_TYPE="${ENV_TYPE:-simple_tag_v3}"
N_ADVERSARIES="${N_ADVERSARIES:-}"

if [[ "$ENV_TYPE" == "simple_tag_v3" ]]; then
  if [[ -z "$N_ADVERSARIES" ]]; then
    echo "Please set N_ADVERSARIES for simple_tag_v3, e.g. N_ADVERSARIES=4 bash run_all_single_scale.sh"
    exit 1
  fi
  NUM_AGENT=$((N_ADVERSARIES + 1))
  SCALE_TAG="${NUM_AGENT}agent"
elif [[ "$ENV_TYPE" == "grf" ]]; then
  GRF_N_LEFT="${GRF_N_LEFT:-2}"
  GRF_N_RIGHT="${GRF_N_RIGHT:-0}"
  NUM_AGENT=$((GRF_N_LEFT + GRF_N_RIGHT))
  SCALE_TAG="${NUM_AGENT}agent"
elif [[ "$ENV_TYPE" == "overcooked" ]]; then
  NUM_AGENT=2
  SCALE_TAG="${NUM_AGENT}agent"
else
  echo "Unsupported ENV_TYPE: $ENV_TYPE (choose simple_tag_v3 | grf | overcooked)"
  exit 1
fi

ROOT_RESULTS_DIR="${ROOT_RESULTS_DIR:-/path/of/results}"
if [[ -z "${EXPERT_CSV:-}" ]]; then
  if [[ "$ENV_TYPE" == "simple_tag_v3" ]]; then
    EXPERT_CSV="$ROOT_RESULTS_DIR/simple_tag_v3/eval/adv_${N_ADVERSARIES}/il_transitions_adv_${N_ADVERSARIES}.csv"
  else
    echo "Please set EXPERT_CSV for ENV_TYPE=$ENV_TYPE"
    exit 1
  fi
fi
TRIAL="${TRIAL:-$((90000 + NUM_AGENT))}"
DEVICE="${DEVICE:-cuda}"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/magail/peddingzoo_${ENV_TYPE}_${SCALE_TAG}_ckpts}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/HySteal_model_${ENV_TYPE}_${SCALE_TAG}/model_pkl_csv}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/HySteal_model_${ENV_TYPE}_${SCALE_TAG}/log_csv}"
DYNAMICS_CKPT="${DYNAMICS_CKPT:-$OUT_DIR/dynamics_best.pt}"

START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-5}"

mkdir -p "$OUT_DIR" "$MODEL_DIR" "$LOG_DIR"

AGENT_ORDER=""
if [[ "$ENV_TYPE" == "simple_tag_v3" ]]; then
  for ((i=0; i<N_ADVERSARIES; i++)); do
    if [[ -z "$AGENT_ORDER" ]]; then
      AGENT_ORDER="adversary_${i}"
    else
      AGENT_ORDER="${AGENT_ORDER},adversary_${i}"
    fi
  done
  AGENT_ORDER="${AGENT_ORDER},agent_0"
fi

run_stage() {
  local s="$1"
  [[ "$s" -ge "$START_STAGE" && "$s" -le "$END_STAGE" ]]
}

echo "[info] env=${ENV_TYPE} scale=${SCALE_TAG} N_ADVERSARIES=${N_ADVERSARIES} NUM_AGENT=${NUM_AGENT}"
echo "[info] expert_csv=${EXPERT_CSV}"
echo "[info] agent_order=${AGENT_ORDER}"
echo "[info] device=${DEVICE}"
echo "[info] stages=${START_STAGE}..${END_STAGE}"

cd magail

if run_stage 1; then
  echo "[stage1] Train BC ensemble..."
  for seed in 0 1 2 3 4 5; do
    ckpt="$OUT_DIR/bc_best_seed${seed}.pth"
    if [[ -f "$ckpt" ]]; then
      echo "  - seed=${seed}: skip (exists: $ckpt)"
      continue
    fi
    sigma="$(python3 - <<PY
seed=$seed
print(f"{0.01 + 0.005*seed:.3f}")
PY
)"
    BC_ARGS=(--csv_path "$EXPERT_CSV" --use_agent_id --epochs 300 --batch_size 1028 --lr 3e-4 --seed "$seed" --device "$DEVICE" --out_dir "$OUT_DIR" --aug_enable --aug_p 1.0 --aug_sigma "$sigma" --alpha_ce 0.0 --lambda_cons 0.3 --cons_type kl_probs)
    if [[ -n "$AGENT_ORDER" ]]; then
      BC_ARGS+=(--agent_order "$AGENT_ORDER")
    fi
    python3 train_bc_from_csv.py \
      "${BC_ARGS[@]}"
  done
fi

if run_stage 2; then
  echo "[stage2] Train dynamics..."
  if [[ -f "$DYNAMICS_CKPT" ]]; then
    echo "  - skip (exists: $DYNAMICS_CKPT)"
  else
    DYN_ARGS=(--csv_path "$EXPERT_CSV" --device "$DEVICE" --out_path "$DYNAMICS_CKPT")
    if [[ -n "$AGENT_ORDER" ]]; then
      DYN_ARGS+=(--agent_order "$AGENT_ORDER")
    fi
    python3 train_dynamics_from_csv.py "${DYN_ARGS[@]}"
  fi
fi

if run_stage 3; then
  echo "[stage3] Gated rollout (dynamics env)..."
  AUG_CSV="$OUT_DIR/aug_gate_${ENV_TYPE}.csv"
  if [[ -f "$AUG_CSV" ]]; then
    echo "  - skip (exists: $AUG_CSV)"
  else
    python3 augment_rollout_with_gate.py \
      --env_name "$ENV_TYPE" \
      --use_dynamics_env \
      --dynamics_ckpt "$DYNAMICS_CKPT" \
      --episodes 500 \
      --keep_steps 20 \
      --ep_len 20 \
      --student_ckpt "$OUT_DIR/bc_best_seed0.pth" \
      --expert_ckpts "$OUT_DIR/bc_best_seed1.pth,$OUT_DIR/bc_best_seed2.pth,$OUT_DIR/bc_best_seed3.pth,$OUT_DIR/bc_best_seed4.pth,$OUT_DIR/bc_best_seed5.pth" \
      --min_vote 0.2 \
      --min_avg_logp -1.8 \
      --out_csv "$AUG_CSV" \
      --device "$DEVICE" \
      --max_attempts 20000000
  fi
fi

if run_stage 4; then
  echo "[stage4] Merge expert + aug..."
  MERGED_CSV="$OUT_DIR/il_merged_expert_aug_${ENV_TYPE}.csv"
  AUG_CSV="$OUT_DIR/aug_gate_${ENV_TYPE}.csv"
  if [[ -f "$MERGED_CSV" ]]; then
    echo "  - skip (exists: $MERGED_CSV)"
  else
    python3 merge_expert_and_aug.py \
      --expert_csv "$EXPERT_CSV" \
      --aug_csv "$AUG_CSV" \
      --out_csv "$MERGED_CSV" \
      --expert_weight 1.0 \
      --aug_weight 0.5 \
      --fill_nan_obs 0.0 \
      --max_t 20
  fi
fi

cd "$ROOT_DIR"

if run_stage 5; then
  echo "[stage5] Train HySteal..."
  python3 magail/main_magail_csv.py \
    --train_mode True \
    --config_train ./config/config_multi.yml \
    --expert_csv "$OUT_DIR/il_merged_expert_aug_${ENV_TYPE}.csv" \
    --num_agent "$NUM_AGENT" \
    --trial "$TRIAL" \
    --eval_model_epoch 2 \
    --save_model_epoch 50 \
    --save_model_path "$MODEL_DIR" \
    --log_root "$LOG_DIR" \
    --plot \
    --bc_pretrain_epochs 50 \
    --bc_batch_size 1000 \
    --bc_lr 3e-4
fi

echo "[done] ${SCALE_TAG} pipeline complete."
