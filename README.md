# Lane Binary Classifier

Lightweight lane in/out binary classifier for autonomous driving perception coursework.

The model receives a front-view road ROI image and predicts whether the ROI is inside the current lane area:

- `in_lane`: the ROI is centered on the current lane area.
- `out_lane`: the ROI is shifted away from the current lane area.

This project includes data conversion, training, evaluation, TensorBoard logging, ONNX export, and ONNX Runtime inference.

## Results

The final model was trained on binary ROI samples converted from the TuSimple lane dataset.

| Split | Accuracy | Precision | Recall | F1 | Samples |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9085 | 0.8254 | 0.9246 | 0.8722 | 1060 |
| Test | 0.9665 | 0.9301 | 0.9732 | 0.9511 | 1224 |

Test confusion matrix:

| Metric | Count | Meaning |
|---|---:|---|
| TP | 399 | true `in_lane`, predicted `in_lane` |
| TN | 784 | true `out_lane`, predicted `out_lane` |
| FP | 30 | true `out_lane`, predicted `in_lane` |
| FN | 11 | true `in_lane`, predicted `out_lane` |

ONNX Runtime examples:

- `in_lane`: `probability_in_lane=0.9432`
- `out_lane`: `probability_in_lane=0.0000`

## Project Structure

```text
lane_binary_classifier/
  configs/
    baseline.json
  docs/
    dataset_preparation.md
    experiment_plan.md
    github_upload.md
    model_design.md
  scripts/
    make_toy_dataset.py
    prepare_tusimple_binary.py
  src/lane_binary_classifier/
    data.py
    evaluate.py
    export_onnx.py
    metrics.py
    model.py
    predict.py
    predict_onnx.py
    train.py
  tests/
  requirements.txt
  SUBMISSION.md
```

Large generated files are intentionally excluded from Git:

```text
data/
outputs/
checkpoints/
*.pt
*.onnx
.venv/
```

## Environment

Recommended:

```text
Python 3.12
```

The code supports Python `>=3.10`.

Install runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Runtime dependencies:

```text
torch
numpy
pillow
tqdm
tensorboard
onnx
onnxruntime
```

## Dataset Format

The training script expects an ImageFolder-style binary classification dataset:

```text
data/tusimple_binary/
  train/
    in_lane/
    out_lane/
  val/
    in_lane/
    out_lane/
  test/
    in_lane/
    out_lane/
```

## Convert TuSimple Labels

Expected raw TuSimple layout:

```text
D:\archive\TUSimple\train_set\
  clips\
  label_data_0313.json
  label_data_0531.json
  label_data_0601.json
```

Convert training split:

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\archive\TUSimple\train_set --label-file D:\archive\TUSimple\train_set\label_data_0313.json --split train --output data\tusimple_binary
```

Convert validation split:

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\archive\TUSimple\train_set --label-file D:\archive\TUSimple\train_set\label_data_0531.json --split val --output data\tusimple_binary
```

Convert test split:

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\archive\TUSimple\train_set --label-file D:\archive\TUSimple\train_set\label_data_0601.json --split test --output data\tusimple_binary
```

The converter estimates the current lane center from TuSimple lane-line annotations. A center crop becomes an `in_lane` sample, while left/right shifted crops become `out_lane` samples.

## Train

```powershell
python -m lane_binary_classifier.train --data-dir data\tusimple_binary --output-dir outputs\tusimple_run --epochs 20 --batch-size 32 --tensorboard
```

Outputs:

```text
outputs/tusimple_run/
  best.pt
  last.pt
  run_config.json
  tensorboard/
```

`best.pt` is selected by validation F1. In the submitted experiment, the best checkpoint was from epoch 18.

## View TensorBoard

```powershell
tensorboard --logdir outputs\tusimple_run\tensorboard
```

Open:

```text
http://localhost:6006/
```

Recorded metrics:

- loss
- accuracy
- precision
- recall
- F1

## Evaluate

```powershell
python -m lane_binary_classifier.evaluate --checkpoint outputs\tusimple_run\best.pt --data-dir data\tusimple_binary --split test --output outputs\tusimple_run\test_metrics.json
```

## Export ONNX

```powershell
python -m lane_binary_classifier.export_onnx --checkpoint outputs\tusimple_run\best.pt --output outputs\tusimple_run\lane_binary_classifier.onnx
```

## ONNX Runtime Inference

```powershell
python -m lane_binary_classifier.predict_onnx --onnx-model outputs\tusimple_run\lane_binary_classifier.onnx --image data\tusimple_binary\test\in_lane\clips_0601_1494452385593783358_20_center.png
```

Example output:

```text
class=in_lane probability_in_lane=0.9432 logit=2.8098 provider=CPUExecutionProvider
```

## Data Augmentation

Training images use simple augmentations to improve generalization:

- horizontal flip
- brightness jitter
- contrast jitter

Validation and test images are not augmented.

## Notes

The submitted package also includes:

- Word experiment report
- PPT design document
- exported ONNX model
- test metrics JSON
- run configuration JSON

See `SUBMISSION.md` for the exact commands and dependency summary.
