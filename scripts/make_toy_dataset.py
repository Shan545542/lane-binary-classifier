from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small synthetic lane ROI dataset.")
    parser.add_argument("--output", default="data/toy_lane")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--val-size", type=int, default=80)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def draw_lane_sample(width: int, height: int, in_lane: bool) -> Image.Image:
    road_color = random.randint(45, 75)
    image = Image.new("RGB", (width, height), (road_color, road_color, road_color + 5))
    draw = ImageDraw.Draw(image)

    horizon_y = int(height * random.uniform(0.15, 0.28))
    draw.rectangle((0, 0, width, horizon_y), fill=(80, 105, 120))

    center = width // 2 if in_lane else random.choice([int(width * 0.22), int(width * 0.78)])
    center += random.randint(-8, 8)
    bottom_half_width = random.randint(38, 50)
    top_half_width = random.randint(8, 15)
    lane_color = tuple(random.randint(190, 245) for _ in range(3))

    left_bottom = (center - bottom_half_width, height)
    left_top = (center - top_half_width, horizon_y)
    right_bottom = (center + bottom_half_width, height)
    right_top = (center + top_half_width, horizon_y)

    draw.line((left_bottom, left_top), fill=lane_color, width=random.randint(2, 4))
    draw.line((right_bottom, right_top), fill=lane_color, width=random.randint(2, 4))

    for _ in range(random.randint(15, 35)):
        x = random.randrange(width)
        y = random.randrange(horizon_y, height)
        v = random.randint(-15, 15)
        base = max(0, min(255, road_color + v))
        draw.point((x, y), fill=(base, base, base))

    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
    return image


def write_split(root: Path, split: str, count: int, width: int, height: int) -> None:
    per_class = count // 2
    for class_name, in_lane in [("in_lane", True), ("out_lane", False)]:
        class_dir = root / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(per_class):
            image = draw_lane_sample(width, height, in_lane=in_lane)
            image.save(class_dir / f"{idx:04d}.png")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.output)
    write_split(root, "train", args.train_size, args.width, args.height)
    write_split(root, "val", args.val_size, args.width, args.height)
    print(f"Wrote toy dataset to {root}")


if __name__ == "__main__":
    main()
