from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lane_binary_classifier.data import ImageConfig, LaneImageFolder, describe_class_balance
from lane_binary_classifier.metrics import BinaryMetricMeter
from lane_binary_classifier.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained lane binary classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--data-dir", required=True, help="Dataset root containing train/val/test.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    meter = BinaryMetricMeter()
    for images, targets in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        meter.update(logits, targets, loss)

    metrics = meter.compute()
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tp": meter.tp,
        "tn": meter.tn,
        "fp": meter.fp,
        "fn": meter.fn,
        "samples": meter.sample_count,
    }


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint.get("image_config", {"width": 160, "height": 96})
    image_config = ImageConfig(width=cfg["width"], height=cfg["height"])

    dataset = LaneImageFolder(Path(args.data_dir) / args.split, image_config=image_config, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint.get("model_name", "lane_cnn"))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "device": str(device),
        "class_balance": describe_class_balance(dataset.samples),
        "metrics": evaluate(model, loader, nn.BCEWithLogitsLoss(), device),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
