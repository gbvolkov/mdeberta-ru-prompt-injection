# Полный цикл обучения от mDeBERTa до V16

## Главный принцип пакета

В пакет **не включены веса промежуточных моделей**.

Обучение начинается с:

```text
microsoft/mdeberta-v3-base
```

и последовательно создаёт:

```text
V10 -> V11 -> V12 -> V13 -> V16
```

Все создаваемые модели записываются только в:

```text
work/models/
```

На момент передачи `work/models/` пуст. Проверка пакета завершается ошибкой, если вне `work/` обнаружены `model.safetensors`, `pytorch_model.bin`, optimizer или checkpoint files.

## Почему цепочка выглядит именно так

```text
V10: fresh training from microsoft/mdeberta-v3-base
V11: fine-tune from V10
V12: critical-recall correction from V11
V13: critical Russian correction from V12
V16: recall restoration from V13
```

V14 и V15 не являются родителями V16. V15 evaluation results использовались только при подготовке уже сохранённого V16 dataset.

## Включённые training datasets

| Stage | Dataset | Train | Validation | Attack train | Benign train |
|---|---|---:|---:|---:|---:|
| V10 | `training-dataset-v10-benign-coverage` | 127,408 | 22,485 | 27,591 | 99,817 |
| V11 | `training-dataset-v11-fp-correction-windowed-200k` | 200,000 | 25,000 | 70,000 | 130,000 |
| V12 | создаётся в `work/datasets` | около 29,970 | около 4,999 | определяется builder | определяется builder |
| V13 | `training-dataset-v13-critical-russian-correction-windowed` | 35,382 | 4,000 | 25,066 | 10,316 |
| V16 | `training-dataset-v16-critical-recall-restoration-windowed` | 37,675 | 6,000 | 23,436 | 14,239 |

V10, V11, V13 и V16 сохранены как готовые Hugging Face `DatasetDict`. Они не перестраиваются при полном training cycle.

Локальная копия исторического V12 `DatasetDict` не сохранилась. Поэтому V12 dataset строится штатным архивным builder из включённых V11 dataset и carrier corpus. Это делает цикл исполнимым от base model, но не даёт права обещать битовое совпадение с историческими V12/V16 weights.

## Дополнительные dataset inputs

В `datasets/raw/` включены:

- V10 phrase banks и label judgments;
- V11 false-positive corpus и полный V10 review output;
- V11 document split manifest;
- benign dev/test и malicious dev/test corpora;
- carrier corpus для V12;
- V13/V16 diagnostic corpora и V13/V15 score outputs, необходимые builders;
- audit sidecars исходных dataset builds.

## Scripts

`scripts/` содержит:

```text
build_training_dataset.py
build_v9_coverage_dataset.py
build_v10_benign_coverage_dataset.py
build_false_positive_corpus.py
run_false_positive_review.py
build_v11_windowed_dataset.py
build_v12_correction_dataset.py
build_v12_frozen_eval_suites.py
check_v12_leakage.py
evaluate_v12_checkpoints.py
build_v13_critical_correction_dataset.py
build_v15_anchored_critical_correction_dataset.py
build_v16_critical_recall_restoration_dataset.py
train_mdeberta_ru_prompt_injection_option_b.py
run_blind_broad_eval.py
run_core_validation.py
```

Builders V10/V11/V13/V16 включены для provenance и повторной подготовки данных. Полный training cycle использует сохранённые prepared datasets для этих stages и строит только отсутствующий V12 dataset.

Сохранённый V13 DatasetDict является authoritative input для training stage. Доступная сейчас версия V13 builder в repository не совпадает по служебным `source_name` с этим историческим DatasetDict, поэтому full-cycle command намеренно не перестраивает V13.

Машиночитаемая lineage и hyperparameters находятся в `configs/training-lineage.json`.

## Установка

```bash
uv sync
```

Base model V10 и teacher для distillation загружаются с Hugging Face при первом запуске:

```text
microsoft/mdeberta-v3-base
protectai/deberta-v3-base-prompt-injection-v2
```

Локальные intermediate models не требуются.

Для CUDA должна быть установлена совместимая CUDA-сборка PyTorch.

## Проверка пакета до обучения

Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\commands\windows\00_verify_package.ps1
```

Linux:

```bash
bash commands/linux/00_verify_package.sh
```

Ожидаемый результат:

```text
status: pass
bundled_model_artifacts: []
```

## Полный цикл одной командой

### Linux / pod

```bash
bash commands/linux/90_run_full_cycle_cuda.sh
```

### Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\commands\windows\90_run_full_cycle_cuda.ps1
```

Команда последовательно выполняет все stages и прекращает работу при первой ошибке.

## Stage 1: V10 от microsoft/mdeberta-v3-base

```text
parent: microsoft/mdeberta-v3-base
teacher: protectai/deberta-v3-base-prompt-injection-v2
dataset: training-dataset-v10-benign-coverage
epochs: 3
learning rate: 1e-5
distillation weight: 0.02
trainable encoder layers: 12
train batch: 16
gradient accumulation: 2
precision: BF16
```

Windows:

```powershell
.\commands\windows\10_train_v10_cuda.ps1
```

Linux:

```bash
bash commands/linux/10_train_v10_cuda.sh
```

## Stage 2: V11 от V10

```text
dataset: training-dataset-v11-fp-correction-windowed-200k
epochs: 1
learning rate: 3e-6
distillation: disabled
trainable encoder layers: last 4
train batch: 32
precision: BF16
```

Windows:

```powershell
.\commands\windows\20_train_v11_cuda.ps1
```

Linux:

```bash
bash commands/linux/20_train_v11_cuda.sh
```

## Stage 3: V12 dataset и модель

Сначала строится отсутствующий V12 dataset:

```powershell
.\commands\windows\30_build_v12_dataset.ps1
```

```bash
bash commands/linux/30_build_v12_dataset.sh
```

Затем обучается V12:

```text
parent: generated V11
learning rate: 2e-6
epochs: 1
last layers: 4
precision: BF16
checkpoint interval: 250
```

```powershell
.\commands\windows\31_train_v12_cuda.ps1
```

```bash
bash commands/linux/31_train_v12_cuda.sh
```

Trainer автоматически загружает лучший checkpoint по validation F1 перед сохранением root model. В историческом цикле V12 release соответствовал checkpoint 750.

## Stage 4: V13 от V12

```text
dataset: training-dataset-v13-critical-russian-correction-windowed
learning rate: 1e-6
epochs: 1
last layers: 4
precision: BF16
checkpoint interval: 250
```

```powershell
.\commands\windows\40_train_v13_cuda.ps1
```

```bash
bash commands/linux/40_train_v13_cuda.sh
```

## Stage 5: V16 от V13

Исторический V16 `run_config.json` фиксирует CPU/FP32:

```powershell
.\commands\windows\50_train_v16_exact_cpu.ps1
```

```bash
bash commands/linux/50_train_v16_exact_cpu.sh
```

Для полного CUDA cycle используется эквивалент с теми же dataset/hyperparameters, но BF16:

```powershell
.\commands\windows\51_train_v16_cuda.ps1
```

```bash
bash commands/linux/51_train_v16_cuda.sh
```

Параметры V16:

```text
dataset: training-dataset-v16-critical-recall-restoration-windowed
learning rate: 8e-7
epochs: 1
distillation: disabled
last layers: 4
checkpoint interval: 250
```

## Validation

```powershell
.\commands\windows\60_validate_v16.ps1 -Device cuda
```

```bash
bash commands/linux/60_validate_v16.sh cuda
```

Validation сравнивает созданные V13 и V16 на historical core diagnostic suite. Primary threshold historical suite: `0.82`.

## Outputs

```text
work/models/mdeberta-ru-prompt-injection-v10-benign-scratch
work/models/mdeberta-ru-prompt-injection-v11-fp-correction-ft
work/models/mdeberta-ru-prompt-injection-v12-critical-correction-ft
work/models/mdeberta-ru-prompt-injection-v13-critical-correction-ft
work/models/mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft
```

Итоговая V16 model:

```text
work/models/mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft
```

## Ограничения воспроизводимости

- В package нет intermediate model weights.
- Base и teacher загружаются с Hugging Face.
- Исторический V12 prepared dataset и frozen V12 eval suite отсутствуют в локальном архиве; V12 dataset создаётся повторно архивным builder.
- CUDA, PyTorch, Transformers и nondeterministic kernels могут изменить точные веса и checkpoint ranking.
- Поэтому package воспроизводит полную архитектуру training cycle, datasets V10/V11/V13/V16 и исторические hyperparameters, но не обещает SHA-identical final weights.
- Diagnostic corpora, использованные при dataset mining, не являются blind acceptance data.
