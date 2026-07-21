# Aerial_3: CLIP-H + Voxel JEPA World Model

This directory contains the policy, evaluator, launcher, tests, and AirSim
transport code. It does not import code from the legacy `UAV-ON/UAV-ON`
directory. Large read-only resources are shared between version directories:

- dataset: `../DATASETS`;
- Unreal environments: `../TEST_ENVS`;
- Hugging Face model cache: `../.cache/huggingface`.

## Main behavior

- all four depth cameras produce direction-specific action safety estimates;
- unsafe translations are rejected by a final safety shield;
- horizontal motion is fixed at 2 m, vertical motion at 1 m, and rotation at 15 degrees;
- directional motion is safe only when clearance exceeds the step plus 0.5 m;
- target matching uses prompt ensembling and localized image crops;
- side-camera detections cause rotation before forward approach;
- search uses visited-cell memory and periodic short scans;
- stopping requires repeated visual evidence and target-region depth;
- decisions and depth diagnostics are saved for every task.
- four depth cameras are projected into a persistent 3D occupancy memory;
- a LeWM-inspired action-conditioned JEPA learns from unlabeled executed
  transitions and autoregressively rolls out discrete action sequences;
- receding-horizon beam search combines collision risk, coverage, latent
  novelty, uncertainty, and the existing CLIP direction prior;
- the world model cannot trigger `stop` or bypass the original depth shield.

The implementation follows the encoder/action-encoder/predictor/rollout
interface of [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), but does
not load its task-specific robot checkpoints because their observations and
actions do not match AirSim UAV navigation. See `log_1/design.json` for the
full architecture, loss, safety boundary, and evaluation protocol.

## Ten-round evaluation result

All ten 47-task rounds were completed. No round reached the requested early
stop threshold (strict success >30% and OSR >50%). The best safe round was
`log_5`: 9/47 strict successes (19.15%), 15/47 OSR (31.91%), and zero physical
collisions. The complete machine-readable comparison is in
`automation_summary.json`; every `log_N` also contains its design and metric
analysis.

To reproduce the selected best round from its learned checkpoint:

```bash
EVAL_SAVE_PATH="$PWD/Aerial_3/reproduction_log5" \
AUTO_START_SERVER=false bash Aerial_3/scripts/eval_cliph.sh \
  --jepa_checkpoint_path "$PWD/Aerial_3/checkpoints/voxel_jepa_log5_best.pt"
```

The launcher defaults have been restored to the selected log-5 policy
settings. Later ablations remain recorded in logs 6-10 but are not the default.

## Run: start the server before the client

Use two terminals. Both commands below are run from the workspace root
`/villa/ftt/UAV-ON`.

### 1. Start the AirVLN server

In terminal 1:

```bash
cd /villa/ftt/UAV-ON
conda activate uavon
python -u Aerial_3/airsim_plugin/AirVLNSimulatorServerTool.py \
  --port 30011 \
  --root_path "$PWD/TEST_ENVS" \
  --gpus 0 \
  --cpu_affinity 0-7
```

Wait until the server prints:

```text
start listening 127.0.0.1:30011
```

Keep terminal 1 running during the complete evaluation.

### 2. Start the evaluation client

After the server is listening, run in terminal 2:

```bash
cd /villa/ftt/UAV-ON
AUTO_START_SERVER=false GPU_ID=0 bash Aerial_3/scripts/eval_cliph.sh
```

`AUTO_START_SERVER=false` makes the launcher require the server started in
terminal 1, so it cannot accidentally start a duplicate service. The client
connects to port 30011 and runs the `Aerial_3` package. Default output is
`Aerial_3/logs`. RGBD saving is disabled unless `--image_save_path PATH` is
supplied.

After evaluation finishes, press `Ctrl+C` in terminal 1 to stop the server.

Important thresholds remain CLI-configurable:

```bash
AUTO_START_SERVER=false bash Aerial_3/scripts/eval_cliph.sh \
  --safety_margin 0.5 \
  --stop_score 0.26 \
    --stop_depth 18
```

The inherited stop-only spatial refinement remains an opt-in experiment:

```bash
EVAL_SAVE_PATH="$PWD/Aerial_3/spatial_reproduction" \
AUTO_START_SERVER=false bash Aerial_3/scripts/eval_cliph.sh \
  --spatial_stop_enabled true \
  --min_stop_depth_support 0.20
```

It subdivides the best semantic crop into a 3-by-3 overlapping grid and
rechecks CLIP agreement and depth support inside the same smaller region.  It
is disabled by default because the Aerial_2 round-11 evaluation did not beat
its log-8 baseline.

Each task stores the legacy-compatible `trajectory.jsonl` plus
`clip_decisions.jsonl`, containing per-camera CLIP scores, target depth,
directional clearances, policy mode, and the reason for every action.

Disable the world model for a copied-policy control run:

```bash
AUTO_START_SERVER=false bash Aerial_3/scripts/eval_cliph.sh \
  --world_model_enabled false
```

## Shared-resource overrides

Every shared path and runtime choice can be overridden without editing code:

```bash
DATASET_PATH=/path/to/WesternTown.json \
ENVIRONMENT_ROOT=/path/to/TEST_ENVS \
HF_HOME=/path/to/huggingface \
PYTHON_BIN=/path/to/python \
GPU_ID=0 \
AUTO_START_SERVER=false \
bash Aerial_3/scripts/eval_cliph.sh
```

Set `AUTO_START_SERVER=false` only when an AirVLN server is already listening
on `SIMULATOR_TOOL_PORT`. Set `EVAL_SAVE_PATH` to keep named experiment runs.

CUDA is required by default so unavailable GPU access cannot silently cause a
slow CPU evaluation. For intentional CPU diagnostics only, pass
`--allow_cpu true`.
