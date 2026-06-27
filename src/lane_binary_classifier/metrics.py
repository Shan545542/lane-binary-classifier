from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    loss: float


class BinaryMetricMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.loss_sum = 0.0
        self.sample_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        labels = targets.long()

        self.tp += int(((preds == 1) & (labels == 1)).sum().item())
        self.tn += int(((preds == 0) & (labels == 0)).sum().item())
        self.fp += int(((preds == 1) & (labels == 0)).sum().item())
        self.fn += int(((preds == 0) & (labels == 1)).sum().item())
        batch_size = int(labels.numel())
        self.loss_sum += float(loss.item()) * batch_size
        self.sample_count += batch_size

    def compute(self) -> BinaryMetrics:
        total = max(self.sample_count, 1)
        precision_den = max(self.tp + self.fp, 1)
        recall_den = max(self.tp + self.fn, 1)
        precision = self.tp / precision_den
        recall = self.tp / recall_den
        f1_den = max(precision + recall, 1e-12)
        return BinaryMetrics(
            accuracy=(self.tp + self.tn) / total,
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / f1_den,
            loss=self.loss_sum / total,
        )
