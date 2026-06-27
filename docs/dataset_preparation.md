# Dataset Preparation

## Why Convert TuSimple

TuSimple is a lane detection dataset. Its labels describe lane line coordinates in full front-view road images. This project solves a simpler binary classification task:

- `in_lane`: the ROI is centered on the current lane area.
- `out_lane`: the ROI is shifted away from the current lane area.

Therefore, the original lane detection annotations must be converted into image classification samples.

## Expected TuSimple Layout

After downloading and extracting the dataset, keep it outside this Git repository:

```text
D:\datasets\tusimple\train_set\
  clips\
  label_data_0313.json
  label_data_0531.json
  label_data_0601.json
```

The raw dataset is large and should not be committed to GitHub.

## Smoke Conversion

Before converting the full dataset, run a small conversion:

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\datasets\tusimple\train_set --label-file D:\datasets\tusimple\train_set\label_data_0313.json --split train --output data\tusimple_binary_smoke --max-items 100
```

Then open several generated images under:

```text
data\tusimple_binary_smoke\train\in_lane
data\tusimple_binary_smoke\train\out_lane
```

Check whether the center crops look like lane-centered ROIs and the negative crops are clearly shifted.

## Full Conversion

Use one label file for training and another for validation:

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\datasets\tusimple\train_set --label-file D:\datasets\tusimple\train_set\label_data_0313.json --split train --output data\tusimple_binary
```

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\datasets\tusimple\train_set --label-file D:\datasets\tusimple\train_set\label_data_0531.json --split val --output data\tusimple_binary
```

Optionally convert a held-out test split:

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\datasets\tusimple\train_set --label-file D:\datasets\tusimple\train_set\label_data_0601.json --split test --output data\tusimple_binary
```

## Train On Converted Data

```powershell
python -m lane_binary_classifier.train --data-dir data\tusimple_binary --output-dir outputs\tusimple_run --epochs 20 --batch-size 32 --tensorboard
```

Evaluate the best checkpoint:

```powershell
python -m lane_binary_classifier.evaluate --checkpoint outputs\tusimple_run\best.pt --data-dir data\tusimple_binary --split test --output outputs\tusimple_run\test_metrics.json
```

## Export And Deploy Check

```powershell
python -m lane_binary_classifier.export_onnx --checkpoint outputs\tusimple_run\best.pt --output outputs\tusimple_run\lane_binary_classifier.onnx
```

```powershell
python -m lane_binary_classifier.predict_onnx --onnx-model outputs\tusimple_run\lane_binary_classifier.onnx --image data\tusimple_binary\val\in_lane\0000.png
```
