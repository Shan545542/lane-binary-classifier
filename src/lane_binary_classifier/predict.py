from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lane_binary_classifier.data import ImageConfig, TARGET_TO_CLASS
from lane_binary_classifier.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one ROI image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def preprocess(image_path: str | Path, image_config: ImageConfig) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_config.width, image_config.height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(image_config.mean)) / np.array(image_config.std)
    return torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint.get("image_config", {"width": 160, "height": 96})
    image_config = ImageConfig(width=cfg["width"], height=cfg["height"])

    model = build_model(checkpoint.get("model_name", "lane_cnn"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(preprocess(args.image, image_config))
        prob = float(torch.sigmoid(logits)[0].item())

    target = 1 if prob >= args.threshold else 0
    print(f"class={TARGET_TO_CLASS[target]} probability_in_lane={prob:.4f}")


if __name__ == "__main__":
    main()
