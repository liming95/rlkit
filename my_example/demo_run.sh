#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="TinyContinuousCartPole-v0"
EPOCHS=5
STEPS_PER_EPOCH=1000
INITIAL_RANDOM_STEPS=1000
EVAL_EPISODES=5
MAX_EPISODE_STEPS=200
OUTPUT_DIR="runs"
EXPERIMENT_NAME="sac_cartpole"
RENDER_EVAL=0
RENDER_INFER=0
PLOT_PROGRESS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --steps-per-epoch)
      STEPS_PER_EPOCH="$2"
      shift 2
      ;;
    --initial-random-steps)
      INITIAL_RANDOM_STEPS="$2"
      shift 2
      ;;
    --eval-episodes)
      EVAL_EPISODES="$2"
      shift 2
      ;;
    --max-episode-steps)
      MAX_EPISODE_STEPS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --render-eval)
      RENDER_EVAL=1
      shift
      ;;
    --render-infer)
      RENDER_INFER=1
      shift
      ;;
    --plot-progress)
      PLOT_PROGRESS=1
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: ./demo_run.sh [options]

Options:
  --env NAME                 Environment name, default: TinyContinuousCartPole-v0
  --epochs N                 Training epochs, default: 5
  --steps-per-epoch N        Exploration/training steps per epoch, default: 1000
  --initial-random-steps N   Random warm-up steps, default: 1000
  --eval-episodes N          Evaluation episodes, default: 5
  --max-episode-steps N      Max path length, default: 200
  --output-dir DIR           Output root, default: runs
  --experiment-name NAME     Experiment folder, default: sac_cartpole
  --render-eval              Render evaluation during training
  --render-infer             Render inference after training
  --plot-progress            Save returns.png
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="${OUTPUT_DIR}/${EXPERIMENT_NAME}/checkpoints/latest.pt"

echo "=== SAC demo: training ==="
echo "env=${ENV_NAME} epochs=${EPOCHS} steps_per_epoch=${STEPS_PER_EPOCH}"
echo "outputs=${OUTPUT_DIR}/${EXPERIMENT_NAME}"

TRAIN_ARGS=(
  "sac.py"
  "--env" "${ENV_NAME}"
  "--epochs" "${EPOCHS}"
  "--steps-per-epoch" "${STEPS_PER_EPOCH}"
  "--initial-random-steps" "${INITIAL_RANDOM_STEPS}"
  "--eval-episodes" "${EVAL_EPISODES}"
  "--max-episode-steps" "${MAX_EPISODE_STEPS}"
  "--output-dir" "${OUTPUT_DIR}"
  "--experiment-name" "${EXPERIMENT_NAME}"
)

if [[ "${PLOT_PROGRESS}" -eq 1 ]]; then
  TRAIN_ARGS+=("--plot-progress")
fi

if [[ "${RENDER_EVAL}" -eq 1 ]]; then
  TRAIN_ARGS+=("--render-eval")
fi

python "${TRAIN_ARGS[@]}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint was not created: ${CHECKPOINT}" >&2
  exit 1
fi

echo
echo "=== SAC demo: saved outputs ==="
echo "progress:   ${OUTPUT_DIR}/${EXPERIMENT_NAME}/progress.csv"
echo "checkpoint: ${CHECKPOINT}"
if [[ "${PLOT_PROGRESS}" -eq 1 ]]; then
  echo "plot:       ${OUTPUT_DIR}/${EXPERIMENT_NAME}/returns.png"
fi

echo
echo "=== SAC demo: inference with saved policy ==="

INFER_ARGS=(
  "infer.py"
  "--checkpoint" "${CHECKPOINT}"
  "--episodes" "3"
  "--max-episode-steps" "${MAX_EPISODE_STEPS}"
)

if [[ "${RENDER_INFER}" -eq 1 ]]; then
  INFER_ARGS+=("--render")
fi

python "${INFER_ARGS[@]}"

echo
echo "=== demo finished ==="
