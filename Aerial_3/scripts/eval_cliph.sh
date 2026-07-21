#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$package_root/.." && pwd)"
dataset_path="${DATASET_PATH:-$workspace_root/DATASETS/valset/DownTown.json}"
environment_root="${ENVIRONMENT_ROOT:-$workspace_root/TEST_ENVS}"
environment_name="${ENVIRONMENT_NAME:-$(basename "$dataset_path" .json)}"
simulator_tool_port="${SIMULATOR_TOOL_PORT:-30011}"
gpu_id="${GPU_ID:-0}"
batch_size="${BATCH_SIZE:-1}"
eval_save_path="${EVAL_SAVE_PATH:-$package_root/log_5_DownTown}"
model_cache="${HF_HOME:-$workspace_root/.cache/huggingface}"
jepa_checkpoint_path="${JEPA_CHECKPOINT_PATH:-$package_root/checkpoints/voxel_jepa_log5_best.pt}"
auto_start_server="${AUTO_START_SERVER:-true}"
cpu_affinity="${CPU_AFFINITY:-0-7}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif [[ -x "$workspace_root/../anaconda3/envs/uavon/bin/python" ]]; then
    python_bin="$workspace_root/../anaconda3/envs/uavon/bin/python"
else
    python_bin="$(command -v python3 || command -v python)"
fi

if [[ ! -f "$dataset_path" ]]; then
    echo "Dataset not found: $dataset_path" >&2
    exit 1
fi
if [[ ! -d "$environment_root/$environment_name" ]]; then
    echo "$environment_name environment not found: $environment_root/$environment_name" >&2
    exit 1
fi
if [[ ! -f "$jepa_checkpoint_path" ]]; then
    echo "Log-5 JEPA checkpoint not found: $jepa_checkpoint_path" >&2
    exit 1
fi
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi

export HF_HOME="$model_cache"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$eval_save_path"

# A single AirSim scene contains one vehicle.  Two evaluators connected to the
# same server port would reset that vehicle underneath each other and silently
# corrupt distance, path length, collision, and success metrics.  Hold a
# process-scoped lock for the complete launcher lifetime; flock releases it
# automatically on every normal or abnormal exit.
lock_path="/tmp/aerial_3_eval_${simulator_tool_port}.lock"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "Another Aerial_3 evaluation is already using AirSim port $simulator_tool_port" >&2
    exit 2
fi

"$python_bin" -c \
    'import airsim, cv2, msgpackrpc, numpy, torch, transformers' \
    >/dev/null

port_is_open() {
    "$python_bin" -c \
        'import socket, sys; s=socket.socket(); s.settimeout(0.2); rc=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(rc != 0)' \
        "$simulator_tool_port"
}

server_pid=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

needs_server=true
for argument in "$@"; do
    if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
        needs_server=false
    fi
done

if [[ "$needs_server" == "true" ]] && ! port_is_open; then
    if [[ "$auto_start_server" != "true" ]]; then
        echo "AirVLN server is not listening on port $simulator_tool_port" >&2
        exit 1
    fi
    server_log="$eval_save_path/airsim_server_${simulator_tool_port}.log"
    "$python_bin" -u "$package_root/airsim_plugin/AirVLNSimulatorServerTool.py" \
        --port "$simulator_tool_port" \
        --root_path "$environment_root" \
        --gpus "$gpu_id" \
        --cpu_affinity "$cpu_affinity" \
        >"$server_log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 50); do
        if port_is_open; then
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "AirVLN server failed; see $server_log" >&2
            exit 1
        fi
        sleep 0.2
    done
    if ! port_is_open; then
        echo "AirVLN server did not become ready; see $server_log" >&2
        exit 1
    fi
fi

echo "Aerial package: $package_root"
echo "Dataset: $dataset_path"
echo "Environment root: $environment_root"
echo "Environment name: $environment_name"
echo "Model cache: $HF_HOME"
echo "Log-5 JEPA checkpoint: $jepa_checkpoint_path"
echo "Evaluation output: $eval_save_path"
echo "Python: $python_bin"

cd "$workspace_root"
"$python_bin" -u -m Aerial_3.eval_cliph \
    --maxActions 150 \
    --eval_save_path "$eval_save_path" \
    --dataset_path "$dataset_path" \
    --xOy_step_size 2 \
    --z_step_size 1 \
    --rotateAngle 15 \
    --safety_margin 0.5 \
    --collision_percentile 10 \
    --scan_turns 1 \
    --periodic_scan_turns 1 \
    --search_moves_per_scan 12 \
    --search_translation_budget 32 \
    --recovery_rotation_limit 6 \
    --world_model_enabled true \
    --jepa_planning_horizon 6 \
    --jepa_beam_width 8 \
    --jepa_override_margin 2.0 \
    --jepa_override_min_risk_reduction 0.0 \
    --jepa_hard_collision_enabled false \
    --jepa_latent_novelty_weight 0.1 \
    --jepa_goal_latent_weight 0.0 \
    --jepa_checkpoint_path "$jepa_checkpoint_path" \
    --is_fixed false \
    --gpu_id "$gpu_id" \
    --batchSize "$batch_size" \
    --simulator_tool_port "$simulator_tool_port" \
    "$@"
