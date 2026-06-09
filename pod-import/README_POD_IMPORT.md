# V17 Pod Import Payload

This folder contains the files needed to train V17 on RunPod:

```text
train_mdeberta_ru_prompt_injection_option_b.py
training-dataset-v17-clean-windowed-250k/
training-dataset-v17-clean-windowed-250k-report.json
V17_CLEAN_WINDOWED_DATASET_PROCEDURE.md
docs/runpod_rtx5090_ssh_training_guide.md
```

Create the upload archive from the repository root:

```powershell
tar -czf runpod_mdeberta_v17_payload.tgz -C .\pod-import .
```

Upload to the Pod:

```powershell
scp -P $SshPort -i $KeyPath `
  .\runpod_mdeberta_v17_payload.tgz `
  "root@${PodIP}:/workspace/mdeberta-run/"
```

Extract on the Pod:

```bash
cd /workspace/mdeberta-run
tar -xzf runpod_mdeberta_v17_payload.tgz
```
