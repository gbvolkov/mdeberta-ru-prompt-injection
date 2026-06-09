# RunPod RTX 5090 SSH Training Guide for V17

This guide runs V17 training on one RunPod RTX 5090 Pod using SSH/SCP and `tmux`.

V17 is trained from the base model:

```text
microsoft/mdeberta-v3-base
```

using the completed clean windowed dataset:

```text
training-dataset-v17-clean-windowed-250k
```

Do not use V10-V16 prepared datasets as training inputs for this run.

## Current V17 Dataset

The local V17 dataset build passed with:

```text
dataset dir:      training-dataset-v17-clean-windowed-250k
validation dir:   training-dataset-v17-clean-windowed-250k-validation
report:           training-dataset-v17-clean-windowed-250k-report.json

total rows:       250,000
train rows:       224,964
validation rows:   25,036
attack rows:      132,726
benign rows:      117,274

underfilled:      none
hard failures:    none
split leakage:    0
positive rows without visible attack: 0
benign rows with visible attack:      0
```

Key component counts:

```text
critical_ru_visible_attack_windows: 36,699
embedded_visible_attack_windows:    31,264
benign_actual_fp_hard_negative:      4,865
proper_benign_prod_windows:         34,997
```

## RunPod Settings

Use:

```text
GPU:              1 x NVIDIA GeForce RTX 5090, 32 GB VRAM
Pricing:          On-Demand
Public IP:        required
Container image:  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
Container disk:   40 GB
Volume disk:      100 GB or larger
Volume path:      /workspace
Expose TCP port:  22
HTTP ports:       not required
Pod name:         mdeberta-ru-v17-rtx5090
```

Use SSH over exposed TCP from the RunPod Connect menu. The public SSH port is the external mapped port, not `22`.

## Local Payload Preparation

Run locally in Windows PowerShell:

```powershell
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection

Get-Item .\train_mdeberta_ru_prompt_injection_option_b.py
Get-ChildItem .\training-dataset-v17-clean-windowed-250k
Get-Item .\training-dataset-v17-clean-windowed-250k-report.json

tar -czf runpod_mdeberta_v17_payload.tgz `
  train_mdeberta_ru_prompt_injection_option_b.py `
  training-dataset-v17-clean-windowed-250k `
  training-dataset-v17-clean-windowed-250k-report.json `
  V17_CLEAN_WINDOWED_DATASET_PROCEDURE.md

Get-Item .\runpod_mdeberta_v17_payload.tgz
```

Set your SSH variables after the Pod is running:

```powershell
$PodIP = "<POD_PUBLIC_IP>"
$SshPort = <EXTERNAL_SSH_PORT>
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_runpod"

Test-Path $KeyPath
```

Upload:

```powershell
scp -P $SshPort -i $KeyPath `
  .\runpod_mdeberta_v17_payload.tgz `
  "root@${PodIP}:/workspace/mdeberta-run/"
```

Connect:

```powershell
ssh -p $SshPort -i $KeyPath "root@$PodIP"
```

## Remote Pod Setup

Run remotely:

```bash
mkdir -p /workspace/mdeberta-run /workspace/hf-cache /workspace/mdeberta-run/logs
cd /workspace/mdeberta-run

df -h /workspace
nvidia-smi
```

Verify CUDA/BF16:

```bash
python - <<'PY'
import torch

print("torch          :", torch.__version__)
print("cuda build     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
print("gpu            :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16 supported :", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

x = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
print("bf16 matmul    :", float((x @ x).mean()))
PY
```

Extract payload:

```bash
cd /workspace/mdeberta-run
tar -xzf runpod_mdeberta_v17_payload.tgz
```

Install dependencies without replacing CUDA PyTorch:

```bash
python -m pip install -U pip
python -m pip install -U \
  transformers \
  datasets \
  accelerate \
  scikit-learn \
  sentencepiece \
  protobuf \
  huggingface-hub \
  zstandard \
  safetensors
```

Do not run:

```bash
python -m pip install -U torch
```

Set persistent caches:

```bash
mkdir -p /workspace/hf-cache /workspace/mdeberta-run/logs

cat >> ~/.bashrc <<'EOF'

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false
EOF

source ~/.bashrc
```

Validate the uploaded V17 dataset:

```bash
cd /workspace/mdeberta-run

python - <<'PY'
from datasets import load_from_disk

ds = load_from_disk("./training-dataset-v17-clean-windowed-250k")
print(ds)
print("splits          :", list(ds.keys()))
print("train rows      :", len(ds["train"]))
print("validation rows :", len(ds["validation"]))
print("columns         :", ds["train"].column_names)

assert len(ds["train"]) == 224964
assert len(ds["validation"]) == 25036
assert "text" in ds["train"].column_names
assert "label" in ds["train"].column_names
PY
```

Install `tmux`:

```bash
apt-get update
apt-get install -y tmux
```

## Create the V17 Training Launcher

Run remotely:

```bash
cd /workspace/mdeberta-run

cat > run_training_v17.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mdeberta-run
mkdir -p logs /workspace/hf-cache

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="./mdeberta-ru-prompt-injection-v17-clean-scratch"
DATASET_DIR="./training-dataset-v17-clean-windowed-250k"
LOG_FILE="logs/train_v17_clean_scratch_rtx5090_$(date +%Y%m%d_%H%M%S).log"

if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: output dir already exists: $OUTPUT_DIR" | tee "$LOG_FILE"
  echo "Use a fresh output dir, or resume intentionally with a separate resume script." | tee -a "$LOG_FILE"
  exit 1
fi

test -d "$DATASET_DIR"
test -f ./train_mdeberta_ru_prompt_injection_option_b.py

echo "Training started at: $(date -Is)" | tee "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"
python -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.version.cuda); print("gpu:", torch.cuda.get_device_name(0)); print("bf16:", torch.cuda.is_bf16_supported())' | tee -a "$LOG_FILE"

python train_mdeberta_ru_prompt_injection_option_b.py \
  --device cuda \
  --teacher-device cuda \
  --bf16 \
  --student-model microsoft/mdeberta-v3-base \
  --prepared-dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --learning-rate 1e-5 \
  --epochs 2 \
  --distill-weight 0.0 \
  --skip-teacher \
  --last-n-layers 12 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --gradient-accumulation-steps 2 \
  --checkpoint-steps 500 \
  --save-total-limit 12 \
  --optim adamw_torch_fused \
  --group-by-length \
  --tf32 \
  --torch-num-threads 6 \
  --preflight-steps 2 \
  2>&1 | tee -a "$LOG_FILE"

echo "Training completed at: $(date -Is)" | tee -a "$LOG_FILE"
EOF

chmod +x run_training_v17.sh
```

This launcher fails if the output directory already exists. That prevents accidental reuse of stale stage caches from another dataset.

## Start Training

Run remotely:

```bash
cd /workspace/mdeberta-run
tmux new -s mdeberta-v17
```

Inside `tmux`:

```bash
cd /workspace/mdeberta-run
./run_training_v17.sh
```

Detach without stopping training:

```text
Ctrl+B, then D
```

## Monitoring

GPU:

```bash
watch -n 2 nvidia-smi
```

Latest log:

```bash
cd /workspace/mdeberta-run
LATEST_LOG=$(ls -1t logs/train_v17_clean_scratch_rtx5090_*.log | head -1)
echo "$LATEST_LOG"
tail -f "$LATEST_LOG"
```

Reattach:

```bash
tmux attach -t mdeberta-v17
```

Inspect checkpoints:

```bash
cd /workspace/mdeberta-run
find ./mdeberta-ru-prompt-injection-v17-clean-scratch -maxdepth 2 -type d | sort
du -sh ./mdeberta-ru-prompt-injection-v17-clean-scratch 2>/dev/null || true
```

## Resume After Interruption

If SSH disconnects but the Pod is still running:

```bash
tmux attach -t mdeberta-v17
```

If training stopped and you want to resume from existing checkpoints, create a resume launcher:

```bash
cd /workspace/mdeberta-run

cat > run_training_v17_resume.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mdeberta-run
mkdir -p logs /workspace/hf-cache

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="./mdeberta-ru-prompt-injection-v17-clean-scratch"
DATASET_DIR="./training-dataset-v17-clean-windowed-250k"
LOG_FILE="logs/train_v17_clean_scratch_resume_rtx5090_$(date +%Y%m%d_%H%M%S).log"

python train_mdeberta_ru_prompt_injection_option_b.py \
  --device cuda \
  --teacher-device cuda \
  --bf16 \
  --student-model microsoft/mdeberta-v3-base \
  --prepared-dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --learning-rate 1e-5 \
  --epochs 2 \
  --distill-weight 0.0 \
  --skip-teacher \
  --last-n-layers 12 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --gradient-accumulation-steps 2 \
  --checkpoint-steps 500 \
  --save-total-limit 12 \
  --optim adamw_torch_fused \
  --group-by-length \
  --tf32 \
  --torch-num-threads 6 \
  --preflight-steps 2 \
  2>&1 | tee -a "$LOG_FILE"
EOF

chmod +x run_training_v17_resume.sh
tmux new -s mdeberta-v17-resume
./run_training_v17_resume.sh
```

The resume launcher intentionally does not use `--rebuild-stage-cache` or `--no-trainer-auto-resume`.

## Low-Memory Fallback

Use only if the main command hits CUDA OOM. This preserves effective batch size 32:

```bash
cd /workspace/mdeberta-run

sed \
  -e 's/mdeberta-ru-prompt-injection-v17-clean-scratch/mdeberta-ru-prompt-injection-v17-clean-scratch-low-memory/g' \
  -e 's/--train-batch-size 16/--train-batch-size 8/g' \
  -e 's/--eval-batch-size 64/--eval-batch-size 32/g' \
  -e 's/--gradient-accumulation-steps 2/--gradient-accumulation-steps 4/g' \
  run_training_v17.sh > run_training_v17_low_memory.sh

chmod +x run_training_v17_low_memory.sh
tmux new -s mdeberta-v17-low-memory
./run_training_v17_low_memory.sh
```

## Verify Finished Model

After training completes:

```bash
cd /workspace/mdeberta-run/mdeberta-ru-prompt-injection-v17-clean-scratch
ls -lah

for f in model.safetensors config.json student_eval_metrics.json threshold_recommendations.json run_config.json; do
  if [ -f "$f" ]; then
    echo "OK: $f"
  else
    echo "MISSING: $f"
  fi
done

cat student_eval_metrics.json
printf '\n\n'
cat threshold_recommendations.json
printf '\n'
```

## Package Results

Run remotely:

```bash
cd /workspace/mdeberta-run

tar -czf mdeberta-ru-prompt-injection-v17-clean-scratch.tgz \
  mdeberta-ru-prompt-injection-v17-clean-scratch \
  logs \
  run_training_v17.sh \
  training-dataset-v17-clean-windowed-250k-report.json \
  V17_CLEAN_WINDOWED_DATASET_PROCEDURE.md

ls -lh mdeberta-ru-prompt-injection-v17-clean-scratch.tgz
```

Download locally:

```powershell
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection

scp -P $SshPort -i $KeyPath `
  "root@${PodIP}:/workspace/mdeberta-run/mdeberta-ru-prompt-injection-v17-clean-scratch.tgz" `
  .

Get-Item .\mdeberta-ru-prompt-injection-v17-clean-scratch.tgz

tar -tzf .\mdeberta-ru-prompt-injection-v17-clean-scratch.tgz |
  Select-String "model.safetensors|student_eval_metrics.json|threshold_recommendations.json|run_config.json"
```

Only terminate the Pod after the archive has been downloaded and verified locally.

## Quick Local Command Checklist

```powershell
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection

tar -czf runpod_mdeberta_v17_payload.tgz `
  train_mdeberta_ru_prompt_injection_option_b.py `
  training-dataset-v17-clean-windowed-250k `
  training-dataset-v17-clean-windowed-250k-report.json `
  V17_CLEAN_WINDOWED_DATASET_PROCEDURE.md

$PodIP = "<POD_PUBLIC_IP>"
$SshPort = <EXTERNAL_SSH_PORT>
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_runpod"

scp -P $SshPort -i $KeyPath `
  .\runpod_mdeberta_v17_payload.tgz `
  "root@${PodIP}:/workspace/mdeberta-run/"

ssh -p $SshPort -i $KeyPath "root@$PodIP"
```

## Quick Remote Command Checklist

```bash
mkdir -p /workspace/mdeberta-run /workspace/hf-cache /workspace/mdeberta-run/logs
cd /workspace/mdeberta-run
tar -xzf runpod_mdeberta_v17_payload.tgz

python -m pip install -U pip
python -m pip install -U transformers datasets accelerate scikit-learn sentencepiece protobuf huggingface-hub zstandard safetensors

apt-get update
apt-get install -y tmux

# create run_training_v17.sh from the guide above
tmux new -s mdeberta-v17
./run_training_v17.sh
```
