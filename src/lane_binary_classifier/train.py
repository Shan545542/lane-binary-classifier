from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from lane_binary_classifier.data import ImageConfig, describe_class_balance, make_loaders
from lane_binary_classifier.metrics import BinaryMetricMeter, BinaryMetrics
from lane_binary_classifier.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lane in/out binary classifier.")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/val folders.")
    parser.add_argument("--output-dir", default="outputs/lane_run", help="Directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> BinaryMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    meter = BinaryMetricMeter()
    progress = tqdm(loader, leave=False, desc="train" if is_train else "val")

    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        meter.update(logits.detach(), targets.detach(), loss.detach())
        metrics = meter.compute()
        progress.set_postfix(loss=f"{metrics.loss:.4f}", f1=f"{metrics.f1:.3f}")

    return meter.compute()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: BinaryMetrics,
) -> None:
    checkpoint = {
        "model_name": "lane_cnn",
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics.__dict__,
        "image_config": {"width": args.width, "height": args.height},
        "classes": {"out_lane": 0, "in_lane": 1},
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    image_config = ImageConfig(width=args.width, height=args.height)
    train_loader, val_loader = make_loaders(
        args.data_dir,
        image_config=image_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metadata = {
        "device": str(device),
        "train_balance": describe_class_balance(train_loader.dataset.samples),
        "val_balance": describe_class_balance(val_loader.dataset.samples),
        "args": vars(args),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} train_f1={train_metrics.f1:.3f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f} "
            f"val_precision={val_metrics.precision:.3f} val_recall={val_metrics.recall:.3f} "
            f"val_f1={val_metrics.f1:.3f}"
        )

        if writer is not None:
            for prefix, metrics in [("train", train_metrics), ("val", val_metrics)]:
                writer.add_scalar(f"{prefix}/loss", metrics.loss, epoch)
                writer.add_scalar(f"{prefix}/accuracy", metrics.accuracy, epoch)
                writer.add_scalar(f"{prefix}/precision", metrics.precision, epoch)
                writer.add_scalar(f"{prefix}/recall", metrics.recall, epoch)
                writer.add_scalar(f"{prefix}/f1", metrics.f1, epoch)

        save_checkpoint(output_dir / "last.pt", model, args, epoch, val_metrics)
        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            save_checkpoint(output_dir / "best.pt", model, args, epoch, val_metrics)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
