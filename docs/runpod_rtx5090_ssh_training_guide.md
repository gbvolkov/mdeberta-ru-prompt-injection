# Running `train_mdeberta_ru_prompt_injection_option_b.py` on RunPod with RTX 5090 and SSH/SCP

**Purpose:** Deploy one RunPod RTX 5090 Pod, connect to it through a real SSH terminal, upload the prepared dataset and Python training script through SCP, run training inside `tmux`, monitor it, recover from interruption, and download the resulting trained model.

**Prepared for:** George Volkov
**RunPod workflow:** Full SSH via public IP + SCP
**GPU:** 1 × NVIDIA GeForce RTX 5090, 32 GB VRAM
**Container image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
**Local operating system:** Windows / PowerShell
**Last verified against RunPod documentation:** 2026-05-23

---

## 0. Important choices before you begin

This guide deliberately uses **SSH/SCP**, not JupyterLab and not `runpodctl`.

| Item                         | Required value / decision                                                |
| ---------------------------- | ------------------------------------------------------------------------ |
| RunPod product               | GPU Pod                                                                  |
| GPU                          | `NVIDIA GeForce RTX 5090`, one GPU                                     |
| Pricing type                 | On-Demand                                                                |
| Pod must support             | **Public IP**                                                      |
| Container image              | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`                      |
| Container disk               | 40 GB                                                                    |
| Volume disk                  | 100 GB                                                                   |
| Persistent working directory | `/workspace`                                                           |
| Expose TCP Ports             | `22`                                                                   |
| Expose HTTP Ports            | Leave blank; not required for SSH workflow                               |
| SSH key                      | Already generated locally and public key already added to RunPod account |
| Data upload/download         | `scp` through the external SSH port displayed by RunPod                |

### Why port `22` is configured but you do not normally type port `22` later

Inside the Pod, SSH listens on TCP port `22`. After deployment, RunPod maps it to a public external port, for example:

```text
TCP port   213.173.109.39:13007 -> :22
```

In that example:

- internal Pod port = `22`;
- public IP = `213.173.109.39`;
- public SSH/SCP port = `13007`.

Your local `ssh` and `scp` commands must use the **external port** shown in the Pod's **Connect** menu, not internal port `22`.

### Why use `/workspace`

Keep the dataset, Hugging Face cache, logs, checkpoints, and final model under `/workspace`. RunPod volume-disk data stored there survives a Pod stop/restart, but it is deleted when the Pod is terminated. Container-disk data is cleared when the Pod stops.

---

# Phase A — Prepare your files on Windows before renting the GPU

Do these steps locally before deploying the Pod, so you do not pay for GPU time while preparing an upload.

## Step 1. Open PowerShell

On your Windows computer:

1. Press the Windows key.
2. Type `PowerShell`.
3. Open **Windows PowerShell** or **Terminal → PowerShell**.

All commands in Phase A are executed in this local PowerShell terminal.

## Step 2. Move to your project directory

Execute:

```powershell
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection
```

## Step 3. Confirm that your training script exists

Execute:

```powershell
Get-Item .\train_mdeberta_ru_prompt_injection_option_b.py
```

You must see a file named:

```text
train_mdeberta_ru_prompt_injection_option_b.py
```

## Step 4. Confirm that your prepared dataset exists

Execute:

```powershell
Get-ChildItem .\training-dataset-v10-benign-coverage
```

You should see a saved Hugging Face dataset layout, normally including entries similar to:

```text
dataset_dict.json
train
validation
```

The training run expects the prepared dataset to contain:

```text
train rows:       127,408
validation rows:   22,485
```

## Step 5. Create one compressed upload archive

Execute:

```powershell
tar -czf runpod_mdeberta_payload.tgz `
  train_mdeberta_ru_prompt_injection_option_b.py `
  training-dataset-v10-benign-coverage
```

## Step 6. Verify the archive

Execute:

```powershell
Get-Item .\runpod_mdeberta_payload.tgz
```

You should see the newly created archive and its size.

## Step 7. Locate your SSH private key file

Your public key is already set in RunPod. You need the corresponding **private** key path locally for `ssh` and `scp`.

First list possible keys:

```powershell
Get-ChildItem $env:USERPROFILE\.ssh
```

Typical private-key paths are:

```text
C:\Users\<your-user>\.ssh\id_ed25519
C:\Users\<your-user>\.ssh\id_ed25519_runpod
```

Set a PowerShell variable to your actual private-key path. Use **one** of the following examples, modifying it if your file has a different name:

```powershell
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_runpod"
```

or:

```powershell
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
```

Verify that the file exists:

```powershell
Test-Path $KeyPath
```

Expected output:

```text
True
```

Do not upload or paste this private-key file anywhere.

---

# Phase B — Create the RunPod Pod in the web console

## Step 8. Open the Pods deployment page

In your web browser:

1. Sign in to the RunPod console.
2. In the left navigation, click **Pods**.
3. Click **Deploy**.

## Step 9. Select an RTX 5090 offer that supports a public IP

On the deployment page:

1. Select the GPU deployment option.
2. Search for `RTX 5090`.
3. Choose **NVIDIA GeForce RTX 5090**, with GPU count set to `1`.
4. Choose **On-Demand**, not Spot/Interruptible, for this first training run.
5. In the available machine/offer information, choose an offer that states that a **Public IP** is available or supported.

**Do not continue with an offer that has no public IP.** Full SSH with SCP/SFTP requires public IP support. Without it, RunPod offers only proxied/basic SSH, which can run terminal commands but cannot be used for SCP/SFTP.

## Step 10. Enter a Pod name

Find the Pod name field and enter:

```text
mdeberta-ru-v10-rtx5090
```

## Step 11. Select the container image

In the template/image selection area, select the image option whose container image is exactly:

```text
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

This is the appropriate image from the choices you provided because it includes CUDA 12.8.1 and PyTorch 2.8.0, suitable for the RTX 5090 Blackwell GPU.

Do **not** use these older image alternatives for this run:

```text
runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

## Step 12. Configure storage

On the deployment form, locate the disk-size settings and enter:

| UI field                                                 |          Value |
| -------------------------------------------------------- | -------------: |
| **Container Disk** / **Container Disk Size** |      `40 GB` |
| **Volume Disk** / **Volume Disk Size**       |     `100 GB` |
| **Volume Mount Path**, when shown                  | `/workspace` |

When your dataset contains confidential/internal prompts or security examples, enable:

```text
Encrypt volume
```

All important training data will be placed under `/workspace`, including:

```text
/workspace/mdeberta-run/
/workspace/hf-cache/
/workspace/mdeberta-run/training-dataset-v10-benign-coverage/
/workspace/mdeberta-run/mdeberta-ru-prompt-injection-v10-benign-scratch/
```

## Step 13. Open the template configuration panel

Still on the deployment screen:

1. Find the selected template/container configuration.
2. Click **Edit Template**.

This is the RunPod menu in which exposed ports are configured.

## Step 14. Configure SSH/SCP TCP access

Inside **Edit Template**, locate the field named:

```text
Expose TCP Ports
```

Enter:

```text
22
```

This tells RunPod that the Pod needs direct TCP access to its SSH service.

## Step 15. Do not configure JupyterLab ports

In the same **Edit Template** panel, locate:

```text
Expose HTTP Ports (Max 10)
```

For this SSH-only workflow, leave this field empty unless it already contains ports required by the official template. You do not need port `8888` for this guide.

## Step 16. Do not alter the template start command

When you selected an official RunPod PyTorch template/image, leave its command/entrypoint/start-command fields unchanged. Official RunPod PyTorch templates are intended to provide SSH access already; changing their startup command can prevent the SSH service from starting.

## Step 17. Save template changes

Click **Save** or **Save Template**, depending on the label shown in the console.

Return to the Pod deployment page.

## Step 18. Final check before deployment

Confirm the deployment values:

| Setting           | Required value                                      |
| ----------------- | --------------------------------------------------- |
| GPU               | `1 × NVIDIA GeForce RTX 5090`, 32 GB             |
| Availability      | Public IP supported                                 |
| Pricing           | On-Demand                                           |
| Pod name          | `mdeberta-ru-v10-rtx5090`                         |
| Image             | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container disk    | `40 GB`                                           |
| Volume disk       | `100 GB`                                          |
| Volume mount      | `/workspace`                                      |
| Expose TCP Ports  | `22`                                              |
| Expose HTTP Ports | Blank / unchanged if automatically populated        |

## Step 19. Deploy the Pod

Click **Deploy On-Demand**.

---

# Phase C — Retrieve the real SSH connection details

## Step 20. Wait for the Pod to start

In the RunPod console:

1. Go to **Pods**.
2. Locate `mdeberta-ru-v10-rtx5090`.
3. Wait until it is shown as running/ready.

## Step 21. Open connection details

1. Expand the running Pod.
2. Click **Connect**.
3. Locate the section named **SSH over exposed TCP**.

Do **not** use a connection shown only under ordinary/basic/proxied **SSH** when you want to upload with SCP. Basic SSH does not support SCP/SFTP.

## Step 22. Copy the SSH-over-exposed-TCP command

RunPod should display a command resembling:

```bash
ssh root@213.173.108.12 -p 17445 -i ~/.ssh/id_ed25519
```

The actual IP address and port will be different. Record:

```text
POD_IP_ADDRESS = the IP after root@
SSH_PORT       = the number after -p
```

For example only:

```text
POD_IP_ADDRESS = 213.173.108.12
SSH_PORT       = 17445
```

### If `SSH over exposed TCP` is not shown

Do not upload or install anything yet. One of the following is true:

- you selected an offer without public IP support; or
- TCP port `22` was not exposed in the template; or
- the selected template did not start an SSH daemon.

First check port configuration:

1. Go to **Pods**.
2. Click the **three-dot menu** next to the Pod.
3. Click **Edit Pod**.
4. Locate **Expose TCP Ports**.
5. Ensure it contains `22`.
6. Save and allow the Pod to reset.
7. Open **Connect** again.

Since this is before dataset upload, resetting the Pod does not discard any important training data.

If there is still no **SSH over exposed TCP** entry, terminate the Pod and redeploy onto an RTX 5090 offer with public IP support.

---

# Phase D — Connect from Windows PowerShell through SSH

## Step 23. Set PowerShell variables for this Pod

Return to the local PowerShell window. Set variables using the values shown in RunPod **Connect → SSH over exposed TCP**.

Replace the examples below with your actual values:

```powershell
$PodIP = "213.173.108.12"
$SshPort = 17445
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_runpod"
```

When your private key has another filename, correct `$KeyPath`, for example:

```powershell
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
```

Verify the key path again:

```powershell
Test-Path $KeyPath
```

Expected output:

```text
True
```

## Step 24. Establish the first SSH connection

Execute:

```powershell
ssh -p $SshPort -i $KeyPath "root@$PodIP"
```

At the first connection, you may see a host authenticity prompt such as:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type:

```text
yes
```

You should receive a remote Linux terminal prompt as the `root` user.

### If SSH asks you for a password

Stop and correct key authentication. RunPod key-authenticated SSH should not require a password. Common causes are:

- the public key stored in RunPod is not the pair corresponding to `$KeyPath`;
- a key fingerprint beginning with `SHA256:` was pasted into RunPod instead of the actual `.pub` line;
- the stored public key is missing the leading `ssh-ed25519` token;
- the wrong private-key path was used in `-i`.

Do not continue to upload or train until the SSH connection opens without a password prompt.

---

# Phase E — Prepare the Pod through the SSH terminal

All commands in this phase are typed in the **remote SSH terminal** after you have logged into the Pod.

## Step 25. Confirm that you are connected to the Pod

Execute remotely:

```bash
whoami
hostname
pwd
```

Expected `whoami` result:

```text
root
```

## Step 26. Check `/workspace` storage

Execute remotely:

```bash
df -h /workspace
ls -lah /workspace
```

Confirm that the mounted workspace has approximately the volume capacity you selected.

## Step 27. Create persistent working directories

Execute remotely:

```bash
mkdir -p /workspace/mdeberta-run
mkdir -p /workspace/hf-cache
mkdir -p /workspace/mdeberta-run/logs
mkdir -p /workspace/venvs
cd /workspace/mdeberta-run
pwd
```

Expected final output:

```text
/workspace/mdeberta-run
```

## Step 28. Verify RTX 5090 visibility

Execute remotely:

```bash
nvidia-smi
```

Confirm that the GPU listed is:

```text
NVIDIA GeForce RTX 5090
```

## Step 29. Verify CUDA, PyTorch, and BF16 support

Execute remotely:

```bash
python - <<'PY'
import torch

print("PyTorch version :", torch.__version__)
print("CUDA build      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in this Pod.")

print("GPU name        :", torch.cuda.get_device_name(0))
print("GPU capability  :", torch.cuda.get_device_capability(0))
print("BF16 supported  :", torch.cuda.is_bf16_supported())
print("Compiled arches :", torch.cuda.get_arch_list())

x = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
y = x @ x
print("BF16 CUDA test  : OK", float(y.mean()))
PY
```

Proceed only when you see confirmation equivalent to:

```text
CUDA available  : True
GPU name        : NVIDIA GeForce RTX 5090
BF16 supported  : True
BF16 CUDA test  : OK ...
```

If this fails, do not begin training. Re-check that the deployed image is `cu1281-torch280`, not one of the older CUDA images.

## Step 30. Exit the remote terminal temporarily before running SCP

To upload from your Windows PC, leave the SSH session by executing remotely:

```bash
exit
```

You should return to your local PowerShell prompt.

---

# Phase F — Upload the training archive with SCP

All commands in this phase run on your **local Windows PowerShell** terminal.

## Step 31. Return to the local project directory

Execute locally:

```powershell
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection
```

## Step 32. Verify the local upload archive again

Execute locally:

```powershell
Get-Item .\runpod_mdeberta_payload.tgz
```

## Step 33. Upload the archive to persistent Pod storage

Execute locally:

```powershell
scp -P $SshPort -i $KeyPath `
  .\runpod_mdeberta_payload.tgz `
  "root@${PodIP}:/workspace/mdeberta-run/"
```

Important details:

- The `-P` in `scp` is uppercase.
- `$SshPort` is the external public port from RunPod **Connect → SSH over exposed TCP**, not port `22`.
- The destination is under `/workspace`, so the uploaded archive survives Pod stop/restart.

Wait until SCP reports `100%` transfer completion.

## Step 34. Reconnect to the Pod over SSH

Execute locally:

```powershell
ssh -p $SshPort -i $KeyPath "root@$PodIP"
```

All subsequent commands are executed remotely inside this SSH session unless marked otherwise.

---

# Phase G — Extract and validate the uploaded project

## Step 35. Verify that the archive arrived

Execute remotely:

```bash
cd /workspace/mdeberta-run
ls -lh
```

You must see:

```text
runpod_mdeberta_payload.tgz
```

## Step 36. Extract the archive

Execute remotely:

```bash
cd /workspace/mdeberta-run
tar -xzf runpod_mdeberta_payload.tgz
ls -lah
```

## Step 37. Verify the training script

Execute remotely:

```bash
test -f ./train_mdeberta_ru_prompt_injection_option_b.py \
  && echo "OK: training script found" \
  || echo "ERROR: training script missing"
```

Expected output:

```text
OK: training script found
```

## Step 38. Verify the dataset directory

Execute remotely:

```bash
test -d ./training-dataset-v10-benign-coverage \
  && echo "OK: dataset directory found" \
  || echo "ERROR: dataset directory missing"
```

Expected output:

```text
OK: dataset directory found
```

## Step 39. Inspect the dataset files

Execute remotely:

```bash
find ./training-dataset-v10-benign-coverage -maxdepth 2 | head -40
```

---

# Phase H — Install Python dependencies without breaking CUDA PyTorch

The selected container image already supplies the CUDA-enabled PyTorch build required for RTX 5090. Do **not** install or upgrade `torch`.

## Step 40. Verify Python and pip

Execute remotely:

```bash
python --version
python -m pip --version
```

## Step 41. Install only the additional packages needed by the training script

Execute remotely:

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

Do **not** execute:

```bash
python -m pip install -U torch
```

## Step 42. Re-test CUDA after package installation

Execute remotely:

```bash
python - <<'PY'
import torch
import transformers
import datasets
import accelerate
import sklearn
import sentencepiece
import zstandard

print("torch        :", torch.__version__)
print("cuda build   :", torch.version.cuda)
print("cuda working :", torch.cuda.is_available())
print("gpu          :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16         :", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
print("transformers :", transformers.__version__)
print("datasets     :", datasets.__version__)
print("accelerate   :", accelerate.__version__)
PY
```

Confirm CUDA remains available and the GPU remains RTX 5090.

---

# Phase I — Validate the prepared dataset on the Pod

## Step 43. Load the dataset with Hugging Face Datasets

Execute remotely:

```bash
cd /workspace/mdeberta-run

python - <<'PY'
from datasets import load_from_disk

path = "./training-dataset-v11-fp-correction-windowed-200k"
ds = load_from_disk(path)

print(ds)
print("splits          :", list(ds.keys()))
print("train rows      :", len(ds["train"]))
print("validation rows :", len(ds["validation"]))
print("train columns   :", ds["train"].column_names)
print("valid columns   :", ds["validation"].column_names)
PY
```

Expected counts:

```text
train rows      : 127408
validation rows : 22485
```

Expected columns must include:

```text
text
label
source_name
```

If the dataset cannot be loaded or the required columns are absent, do not start training. Correct the archive or upload the correct prepared dataset first.

---

# Phase J — Configure persistent caches and logs

## Step 44. Set cache environment variables for the current SSH session

Execute remotely:

```bash
mkdir -p /workspace/hf-cache
mkdir -p /workspace/mdeberta-run/logs

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false
```

## Step 45. Persist these variables for reconnects

Execute remotely:

```bash
cat >> ~/.bashrc <<'EOF'

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false
EOF

source ~/.bashrc
```

## Step 46. Verify the values

Execute remotely:

```bash
echo "$HF_HOME"
echo "$HF_DATASETS_CACHE"
echo "$TOKENIZERS_PARALLELISM"
```

Expected output:

```text
/workspace/hf-cache
/workspace/hf-cache/datasets
false
```

---

# Phase K — Install `tmux` and prepare a restart-safe training launch

Training must be run inside `tmux`; otherwise, a dropped SSH session may terminate the foreground training process.

## Step 47. Install `tmux`

Execute remotely:

```bash
apt-get update
apt-get install -y tmux
```

## Step 48. Create a reusable training shell script

Execute remotely exactly as follows:

```bash
cd /workspace/mdeberta-run

cat > run_training.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mdeberta-run
mkdir -p logs /workspace/hf-cache

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false

LOG_FILE="logs/train_rtx5090_$(date +%Y%m%d_%H%M%S).log"

echo "Training started at: $(date -Is)" | tee "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"
python -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.version.cuda); print("gpu:", torch.cuda.get_device_name(0)); print("bf16:", torch.cuda.is_bf16_supported())' | tee -a "$LOG_FILE"

hf download gbv/mdeberta-ru-prompt-injection \
  --revision a672441bb45fa2fb75ca3be019bfd168ae2a8f50 \
  --local-dir ./mdeberta-ru-prompt-injection-v11-fp-correction-ft \
  --max-workers 4

python train_mdeberta_ru_prompt_injection_option_b.py \
  --device cuda \
  --bf16 \
  --student-model ./mdeberta-ru-prompt-injection-v11-fp-correction-ft \
  --prepared-dataset-dir ./training-dataset-v12-russian-critical-correction-windowed \
  --output-dir ./mdeberta-ru-prompt-injection-v12-critical-correction-ft \
  --learning-rate 2e-6 \
  --epochs 1 \
  --distill-weight 0.0 \
  --skip-teacher \
  --last-n-layers 4 \
  --train-batch-size 32 \
  --eval-batch-size 128 \
  --gradient-accumulation-steps 1 \
  --checkpoint-steps 250 \
  --save-total-limit 8 \
  --optim adamw_torch_fused \
  --group-by-length \
  --tf32 \
  --torch-num-threads 6 \
  --preflight-steps 2 \
  2>&1 | tee -a "$LOG_FILE"

echo "Training command completed at: $(date -Is)" | tee -a "$LOG_FILE"
EOF

chmod +x run_training.sh
```

### Intentional differences from your original local command

Your original command used:

```text
--no-trainer-auto-resume
--rebuild-stage-cache
```

The cloud-run script above deliberately omits those two flags. On a new output directory, caches do not exist and will be created normally. On a retry or recovered Pod, omission allows the program to reuse its stage cache and automatically resume from the latest safe Trainer checkpoint, reducing repeated work and GPU cost.

The command uses `python` rather than `uv run python` because the selected Pod image already supplies a verified CUDA/PyTorch runtime. Introducing an additional `uv`-managed environment on the first cloud run can unnecessarily change which PyTorch installation is used.

## Step 49. Start a named `tmux` session

Execute remotely:

```bash
cd /workspace/mdeberta-run
tmux new -s mdeberta-training
```

Your terminal view is now inside `tmux`.

## Step 50. Launch training inside `tmux`

Inside the `tmux` session, execute:

```bash
cd /workspace/mdeberta-run
./run_training.sh
```

The program will begin by loading/scoring/tokenizing data and running its preflight; ordinary Trainer progress output may therefore not appear immediately.

## Step 51. Detach without stopping training

After output has begun, detach from `tmux` by pressing:

```text
Ctrl+B
```

release the keys, then press:

```text
D
```

You return to the normal remote shell while the training process remains active.

---

# Phase L — Monitor training through SSH

You can monitor from the same SSH connection after detaching, or open another local PowerShell terminal and connect again with:

```powershell
ssh -p $SshPort -i $KeyPath "root@$PodIP"
```

## Step 52. Monitor GPU activity

Execute remotely:

```bash
watch -n 2 nvidia-smi
```

Stop watching by pressing:

```text
Ctrl+C
```

## Step 53. Monitor the latest log

Execute remotely:

```bash
cd /workspace/mdeberta-run
LATEST_LOG=$(ls -1t logs/train_rtx5090_*.log | head -1)
echo "$LATEST_LOG"
tail -f "$LATEST_LOG"
```

Stop following the log with `Ctrl+C` without stopping training.

## Step 54. Return to the actual live training screen

Execute remotely:

```bash
tmux attach -t mdeberta-training
```

Detach again with `Ctrl+B`, then `D`.

## Step 55. Inspect created stage caches and checkpoints

Execute remotely:

```bash
cd /workspace/mdeberta-run

find ./mdeberta-ru-prompt-injection-v10-benign-scratch -maxdepth 2 -type d | sort

du -sh ./mdeberta-ru-prompt-injection-v10-benign-scratch 2>/dev/null || true
```

Expected paths after the corresponding stages have run include:

```text
mdeberta-ru-prompt-injection-v10-benign-scratch/stage-cache
mdeberta-ru-prompt-injection-v10-benign-scratch/checkpoint-...
```

---

# Phase M — Recover safely from interruption

## Scenario 1. Your local SSH window closes but the Pod remains running

1. Open a new PowerShell window locally.
2. Reset the variables if necessary:

```powershell
$PodIP = "<IP shown in RunPod Connect>"
$SshPort = <external SSH port shown in RunPod Connect>
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_runpod"
```

3. Reconnect:

```powershell
ssh -p $SshPort -i $KeyPath "root@$PodIP"
```

4. Re-open the live training process:

```bash
tmux attach -t mdeberta-training
```

## Scenario 2. The training process stopped with an error, but the Pod still exists

Check the last log first:

```bash
cd /workspace/mdeberta-run
LATEST_LOG=$(ls -1t logs/train_rtx5090_*.log | head -1)
tail -100 "$LATEST_LOG"
```

When the error is not an out-of-memory error and you are ready to retry, start a new tmux session and rerun the same launcher:

```bash
cd /workspace/mdeberta-run
tmux new -s mdeberta-resume
./run_training.sh
```

Because the launcher omits `--rebuild-stage-cache` and `--no-trainer-auto-resume`, your Python script can reuse matching preprocessing caches and resume from the latest checkpoint.

## Scenario 3. The Pod was stopped and restarted

The Pod's public IP or external mapped SSH port may change after a reset/restart. In the RunPod console:

1. Open **Pods**.
2. Expand your Pod.
3. Click **Connect**.
4. Copy the new details shown under **SSH over exposed TCP**.
5. Update your local `$PodIP` and `$SshPort` variables.
6. Reconnect with SSH.
7. Inspect `/workspace/mdeberta-run` and rerun `./run_training.sh` inside `tmux` if training is no longer active.

## Scenario 4. CUDA out-of-memory failure

With `--train-batch-size 16` on RTX 5090 32 GB, memory should be reasonably conservative. If a CUDA out-of-memory error occurs, do **not** overwrite the original output directory with changed batch settings. Create a low-memory launcher targeting a new output directory:

```bash
cd /workspace/mdeberta-run

sed \
  -e 's/mdeberta-ru-prompt-injection-v10-benign-scratch/mdeberta-ru-prompt-injection-v10-benign-scratch-low-memory/g' \
  -e 's/--train-batch-size 16/--train-batch-size 8/g' \
  -e 's/--eval-batch-size 64/--eval-batch-size 32/g' \
  -e 's/--gradient-accumulation-steps 2/--gradient-accumulation-steps 4/g' \
  -e 's/--teacher-batch-size 16/--teacher-batch-size 8/g' \
  run_training.sh > run_training_low_memory.sh

chmod +x run_training_low_memory.sh

tmux new -s mdeberta-low-memory
./run_training_low_memory.sh
```

The low-memory training geometry still preserves the original effective training batch size:

```text
8 × gradient_accumulation_steps 4 = effective batch size 32
```

---

# Phase N — Verify the finished trained model

After the training process returns successfully, execute remotely:

## Step 56. Enter the output directory

```bash
cd /workspace/mdeberta-run/mdeberta-ru-prompt-injection-v11-fp-correction-ft
ls -lah
```

## Step 57. Confirm essential result files

Execute remotely:

```bash
for f in model.safetensors config.json student_eval_metrics.json threshold_recommendations.json run_config.json; do
  if [ -f "$f" ]; then
    echo "OK: $f"
  else
    echo "MISSING: $f"
  fi
done
```

## Step 58. Read evaluation and threshold outputs

Execute remotely:

```bash
cat student_eval_metrics.json
printf '\n\n'
cat threshold_recommendations.json
printf '\n'
```

## Step 59. Run the generated inference example when it exists

Execute remotely:

```bash
if [ -f inference_example.py ]; then
  python inference_example.py
else
  echo "No inference_example.py file was found; inspect the saved model files manually."
fi
```

---

# Phase O — Package and download results through SCP

## Step 60. Create a compressed results archive on the Pod

Execute remotely:

```bash
cd /workspace/mdeberta-run

tar -czf mdeberta-ru-prompt-injection-v11-fp-correction-ft.tgz \
  mdeberta-ru-prompt-injection-v11-fp-correction-ft \
  logs \
  run_training.sh

ls -lh mdeberta-ru-prompt-injection-v11-fp-correction-fttgz
```

## Step 61. Leave the remote SSH shell

Execute remotely:

```bash
exit
```

You are now back in local PowerShell.

## Step 62. Download the results archive to your Windows project folder

Execute locally:

```powershell
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection

scp -P $SshPort -i $KeyPath `
  "root@${PodIP}:/workspace/mdeberta-run/mdeberta-ru-prompt-injection-v11-fp-correction-ft.tgz" `
  .
```

Wait until SCP completes.

## Step 63. Verify the downloaded archive locally

Execute locally:

```powershell
Get-Item .\mdeberta-ru-prompt-injection-v11-fp-correction-ft.tgz
```

Inspect key contents:

```powershell
tar -tzf .\mdeberta-ru-prompt-injection-v11-fp-correction-ft.tgz |
  Select-String "model.safetensors|student_eval_metrics.json|threshold_recommendations.json|run_config.json"
```

You must see entries for the trained model and metric files before deleting the Pod.

Optional: extract the archive locally:

```powershell
tar -xzf .\mdeberta-ru-prompt-injection-v11-fp-correction-ft.tgz
```

---

# Phase P — Stop billing and remove the Pod safely

## Step 64. Stop the Pod after the result archive has been downloaded

In the RunPod console:

1. Open **Pods**.
2. Expand `mdeberta-ru-v10-rtx5090`.
3. Click the square **Stop** button.
4. Confirm.

Stopping releases the GPU and preserves `/workspace` volume-disk contents, but RunPod continues charging for retained volume storage while the Pod remains stopped.

## Step 65. Terminate the Pod after local verification

Only after Step 63 confirms that the trained model and metric files are present on your Windows computer:

1. In the RunPod console, open **Pods**.
2. Expand the stopped Pod.
3. Click **Terminate** / the trash icon.
4. Confirm.

Termination permanently deletes data on the Pod's volume disk unless you used a separate network volume. Do not terminate first and download later.

---

# Appendix A — Exact menu/field checklist for Pod deployment

| RunPod menu path / field                                     | Value to enter or action                            |
| ------------------------------------------------------------ | --------------------------------------------------- |
| **Pods → Deploy**                                     | Open Pod deployment workflow                        |
| GPU search                                                   | Search `RTX 5090`                                 |
| GPU type                                                     | Select `NVIDIA GeForce RTX 5090`                  |
| GPU count                                                    | `1`                                               |
| Pricing                                                      | `On-Demand`                                       |
| Offer/capacity requirement                                   | Select one that supports**Public IP**         |
| Pod name                                                     | `mdeberta-ru-v10-rtx5090`                         |
| Template/image                                               | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container Disk                                               | `40 GB`                                           |
| Volume Disk                                                  | `100 GB`                                          |
| Volume mount path, if displayed                              | `/workspace`                                      |
| Encrypt volume                                               | Enable for sensitive data                           |
| **Edit Template → Expose TCP Ports**                  | `22`                                              |
| **Edit Template → Expose HTTP Ports (Max 10)**        | Leave blank for SSH-only workflow                   |
| Template start command                                       | Leave unchanged                                     |
| Deploy button                                                | **Deploy On-Demand**                          |
| After start:**Pod → Connect → SSH over exposed TCP** | Copy public IP and external SSH port                |
| File transfer method                                         | `scp` using external port from Connect menu       |

---

# Appendix B — Local Windows PowerShell command checklist

Replace the private key filename when necessary.

```powershell
# Prepare payload before deploying the GPU Pod
cd C:\Projects\guardrails\mdeberta-ru-prompt-injection

Get-Item .\train_mdeberta_ru_prompt_injection_option_b.py
Get-ChildItem .\training-dataset-v10-benign-coverage

tar -czf runpod_mdeberta_payload.tgz `
  train_mdeberta_ru_prompt_injection_option_b.py `
  training-dataset-v10-benign-coverage

# After the Pod is running: enter values from Connect → SSH over exposed TCP
$PodIP = "<POD_PUBLIC_IP>"
$SshPort = <EXTERNAL_SSH_PORT>
$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_runpod"

Test-Path $KeyPath

# Test terminal connection
ssh -p $SshPort -i $KeyPath "root@$PodIP"

# Upload payload
scp -P $SshPort -i $KeyPath `
  .\runpod_mdeberta_payload.tgz `
  "root@${PodIP}:/workspace/mdeberta-run/"

# Reconnect for setup/training
ssh -p $SshPort -i $KeyPath "root@$PodIP"

# After training and after packaging results on the Pod: download results
scp -P $SshPort -i $KeyPath `
  "root@${PodIP}:/workspace/mdeberta-run/mdeberta-ru-prompt-injection-v10-benign-results.tgz" `
  .

# Validate downloaded contents
tar -tzf .\mdeberta-ru-prompt-injection-v10-benign-results.tgz |
  Select-String "model.safetensors|student_eval_metrics.json|threshold_recommendations.json|run_config.json"
```

---

# Appendix C — Remote Pod SSH command checklist

```bash
# Initial workspace preparation
mkdir -p /workspace/mdeberta-run /workspace/hf-cache /workspace/mdeberta-run/logs
cd /workspace/mdeberta-run

df -h /workspace
nvidia-smi

python - <<'PY'
import torch
print("PyTorch version :", torch.__version__)
print("CUDA build      :", torch.version.cuda)
print("CUDA available  :", torch.cuda.is_available())
print("GPU name        :", torch.cuda.get_device_name(0))
print("BF16 supported  :", torch.cuda.is_bf16_supported())
x = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
print("BF16 CUDA test  : OK", float((x @ x).mean()))
PY

# After SCP upload
cd /workspace/mdeberta-run
tar -xzf runpod_mdeberta_payload.tgz

python -m pip install -U pip
python -m pip install -U \
  transformers datasets accelerate scikit-learn sentencepiece \
  protobuf huggingface-hub zstandard safetensors

# Validate dataset
python - <<'PY'
from datasets import load_from_disk
ds = load_from_disk("./training-dataset-v10-benign-coverage")
print(ds)
print("train rows      :", len(ds["train"]))
print("validation rows :", len(ds["validation"]))
print("columns         :", ds["train"].column_names)
PY

# Persistent cache environment
mkdir -p /workspace/hf-cache /workspace/mdeberta-run/logs
cat >> ~/.bashrc <<'EOF'

export HF_HOME=/workspace/hf-cache
export HF_DATASETS_CACHE=/workspace/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false
EOF
source ~/.bashrc

# Process persistence
apt-get update
apt-get install -y tmux

# Use the run_training.sh creation command in Phase K, then launch:
tmux new -s mdeberta-training
./run_training.sh
```

---

# Appendix D — Source references

The workflow above is based on the following RunPod documentation pages:

1. **Connect to a Pod with SSH** — explains basic versus full SSH; full SSH via public IP supports SCP/SFTP, and the Pod Connect tab supplies the `SSH over exposed TCP` command.[https://docs.runpod.io/pods/configuration/use-ssh](https://docs.runpod.io/pods/configuration/use-ssh)
2. **Expose ports** — explains that ports are configured through **Edit Template**, with SSH port `22` entered in **Expose TCP Ports**, and that the Connect menu shows the external public port mapping.[https://docs.runpod.io/pods/configuration/expose-ports](https://docs.runpod.io/pods/configuration/expose-ports)
3. **Transfer files** — provides SCP upload/download syntax over SSH.[https://docs.runpod.io/pods/storage/transfer-files](https://docs.runpod.io/pods/storage/transfer-files)
4. **Storage options** — documents `/workspace` volume-disk persistence across stop/restart, deletion on termination, and the Encrypt volume checkbox.[https://docs.runpod.io/pods/storage/types](https://docs.runpod.io/pods/storage/types)
5. **Manage Pods** — documents the **Pods → Deploy**, **Stop**, **Edit Pod**, **Terminate**, and logs workflows.
   [https://docs.runpod.io/pods/manage-pods](https://docs.runpod.io/pods/manage-pods)
