# Model Design

## Task

The model receives a front-view local ROI image and predicts whether the ego vehicle is inside the target lane area.

## Input And Output

- Input: RGB ROI image, resized to `160 x 96`
- Output: one logit
- Probability: `sigmoid(logit)`
- Class rule: probability greater than or equal to `0.5` is `in_lane`

## Network

`LaneBinaryNet` is a compact CNN:

1. Four convolution blocks extract low-level edges, lane markings, road texture and spatial layout.
2. Adaptive average pooling converts variable spatial responses into a compact feature vector.
3. A dropout layer reduces overfitting.
4. A linear layer produces the binary classification logit.

## Training Objective

The loss function is `BCEWithLogitsLoss`, which combines sigmoid activation and binary cross entropy in a numerically stable form.

## Metrics

The training script reports:

- Accuracy
- Precision
- Recall
- F1 score
- Average loss

F1 is used as the primary checkpoint metric because the two classes may be imbalanced after ROI generation.
