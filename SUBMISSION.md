# Submission Notes

## Python Version

Recommended Python version:

```text
Python 3.12
```

The project code requires Python `>=3.10`.

## Runtime Dependencies

Install the model training and inference dependencies with:

```powershell
pip install -r requirements.txt
pip install -e .
```

Core dependencies:

```text
torch
numpy
pillow
tqdm
tensorboard
onnx
onnxruntime
```

These cover:

- training and validation
- TensorBoard logging
- PyTorch checkpoint inference
- ONNX export
- ONNX Runtime inference

## Optional Documentation Dependencies

The submitted Word report and PPT are already generated. Only install these if the documents need to be regenerated:

```powershell
pip install -r requirements-docs.txt
```

Optional documentation dependencies:

```text
python-docx
python-pptx
```

## Main Commands

Train:

```powershell
python -m lane_binary_classifier.train --data-dir data\tusimple_binary --output-dir outputs\tusimple_run --epochs 20 --batch-size 32 --tensorboard
```

Evaluate:

```powershell
python -m lane_binary_classifier.evaluate --checkpoint outputs\tusimple_run\best.pt --data-dir data\tusimple_binary --split test
```

Export ONNX:

```powershell
python -m lane_binary_classifier.export_onnx --checkpoint outputs\tusimple_run\best.pt --output outputs\tusimple_run\lane_binary_classifier.onnx
```

ONNX Runtime inference:

```powershell
python -m lane_binary_classifier.predict_onnx --onnx-model outputs\tusimple_run\lane_binary_classifier.onnx --image data\tusimple_binary\test\in_lane\clips_0601_1494452385593783358_20_center.png
```
