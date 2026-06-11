# V18 Linux CUDA 20K Dry Run

This folder runs the V18 20K distribution dry run on a Linux CUDA machine.

Required reviewed input files in the working directory:

```text
v18-reviewed-near-boundary-benign-source.jsonl
v18-hard-fn-reviewed-source.jsonl
v18-reviewed-attack-bank-source.jsonl
```

The near-boundary benign file must contain reviewed benign rows with flags such as `manual_reviewed_benign`, `reviewed_benign`, `confirmed_benign`, or `false_positive`.

The hard-FN file must contain visible attack windows with valid `attack_anchor_text`.

The reviewed attack bank must contain reviewed or trusted attack rows. `attack_visible_in_window` alone is not review evidence.

Install on the pod:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -m pip install -r requirements-v18-linux-cuda.txt
```

Run:

```bash
chmod +x run_v18_20k_cuda_linux.sh
./run_v18_20k_cuda_linux.sh
```

Optional environment overrides:

```bash
MODEL_ID=gbv/mdeberta-ru-prompt-injection
PYTHON_BIN=python
OUTPUT_ROOT=./v18-run-linux-cuda-20k
REVIEWED_MINED_BENIGN_JSONL=./v18-reviewed-near-boundary-benign-source.jsonl
HARD_FN_SOURCE_JSONL=./v18-hard-fn-reviewed-source.jsonl
REVIEWED_ATTACK_BANK_JSONL=./v18-reviewed-attack-bank-source.jsonl
FP_TARGET_DOCUMENTS=20000
TARGET_TOTAL_ROWS=20000
VALIDATION_ROWS=2000
```

The script runs:

```text
build_false_positive_corpus.py
run_false_positive_review.py with --device cuda
prepare_v18_external_inputs.py
validate_v18_external_inputs.py
build_v18_clean_500k_dataset.py --dry-run
```
