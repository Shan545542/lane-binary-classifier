from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TuSimple lane labels to binary ROI crops.")
    parser.add_argument("--tusimple-root", required=True, help="Root directory containing TuSimple images.")
    parser.add_argument("--label-file", required=True, help="TuSimple JSON-lines label file.")
    parser.add_argument("--output", default="data/tusimple_binary")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--crop-width", type=int, default=320)
    parser.add_argument("--crop-height", type=int, default=192)
    parser.add_argument("--resize-width", type=int, default=160)
    parser.add_argument("--resize-height", type=int, default=96)
    parser.add_argument("--y-ref", type=int, default=590, help="Reference row near ego vehicle.")
    parser.add_argument("--negative-shift", type=int, default=260)
    parser.add_argument("--max-items", type=int, default=0, help="0 means convert all labels.")
    return parser.parse_args()


def lane_x_at_y(lane: list[int], h_samples: list[int], y_ref: int) -> int | None:
    valid = [(x, y) for x, y in zip(lane, h_samples, strict=False) if x >= 0]
    if not valid:
        return None
    x, _ = min(valid, key=lambda item: abs(item[1] - y_ref))
    return x


def estimate_lane_center(record: dict, image_width: int, y_ref: int) -> int | None:
    xs = []
    for lane in record["lanes"]:
        x = lane_x_at_y(lane, record["h_samples"], y_ref)
        if x is not None:
            xs.append(x)
    xs = sorted(set(xs))
    if len(xs) < 2:
        return None

    image_center = image_width / 2
    left = [x for x in xs if x < image_center]
    right = [x for x in xs if x > image_center]
    if not left or not right:
        return None
    return int((max(left) + min(right)) / 2)


def crop_roi(
    image: Image.Image,
    center_x: int,
    crop_width: int,
    crop_height: int,
    resize_width: int,
    resize_height: int,
) -> Image.Image | None:
    width, height = image.size
    left = center_x - crop_width // 2
    right = center_x + crop_width // 2
    top = height - crop_height
    bottom = height
    if left < 0 or right > width or top < 0:
        return None
    roi = image.crop((left, top, right, bottom))
    return roi.resize((resize_width, resize_height), Image.BILINEAR)


def save_sample(image: Image.Image, output_dir: Path, class_name: str, stem: str, suffix: str) -> None:
    target_dir = output_dir / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    image.save(target_dir / f"{stem}_{suffix}.png")


def main() -> None:
    args = parse_args()
    tusimple_root = Path(args.tusimple_root)
    output_dir = Path(args.output) / args.split
    converted = 0
    skipped = 0

    with Path(args.label_file).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if args.max_items and index >= args.max_items:
                break
            record = json.loads(line)
            image_path = tusimple_root / record["raw_file"]
            if not image_path.exists():
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")
            lane_center = estimate_lane_center(record, image.width, args.y_ref)
            if lane_center is None:
                skipped += 1
                continue

            stem = Path(record["raw_file"]).with_suffix("").as_posix().replace("/", "_")
            positive = crop_roi(
                image,
                lane_center,
                args.crop_width,
                args.crop_height,
                args.resize_width,
                args.resize_height,
            )
            if positive is None:
                skipped += 1
                continue
            save_sample(positive, output_dir, "in_lane", stem, "center")

            for side, direction in [("left", -1), ("right", 1)]:
                negative = crop_roi(
                    image,
                    lane_center + direction * args.negative_shift,
                    args.crop_width,
                    args.crop_height,
                    args.resize_width,
                    args.resize_height,
                )
                if negative is not None:
                    save_sample(negative, output_dir, "out_lane", stem, side)

            converted += 1

    print(f"converted={converted} skipped={skipped} output={output_dir}")


if __name__ == "__main__":
    main()
