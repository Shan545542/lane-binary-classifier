# Experiment Plan

## Data

Use an open-source lane dataset such as TuSimple. Convert official lane annotations into binary ROI crops:

- Positive sample: crop around the estimated current lane center.
- Negative sample: crop shifted to the left or right of the current lane.

For a quick sanity check, use `scripts/make_toy_dataset.py` to generate synthetic lane-like images.

## Training

Suggested baseline:

- Image size: `160 x 96`
- Optimizer: AdamW
- Learning rate: `1e-3`
- Batch size: `32`
- Epochs: `20`
- Augmentation: horizontal flip, brightness jitter, contrast jitter

## Expected Outputs

- `best.pt`: best checkpoint by validation F1
- `last.pt`: final checkpoint
- `run_config.json`: configuration and class balance
- TensorBoard logs under `tensorboard/`

## Report Outline

1. Problem definition
2. Dataset and preprocessing
3. Model architecture
4. Training settings
5. Quantitative results
6. Error analysis
7. Possible improvements
