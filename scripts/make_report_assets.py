from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "report_assets"
TB_DIR = ROOT / "outputs" / "tusimple_run" / "tensorboard"
ONNX_MODEL = ROOT / "outputs" / "tusimple_run" / "lane_binary_classifier.onnx"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_scalars() -> dict[str, list[tuple[int, float]]]:
    accumulator = EventAccumulator(str(TB_DIR))
    accumulator.Reload()
    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]
    return scalars


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[tuple[int, float]], tuple[int, int, int]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x0, y0, x1, y1 = box
    font_title = load_font(22, bold=True)
    font = load_font(15)
    small = load_font(12)
    draw.rounded_rectangle(box, radius=14, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.text((x0 + 18, y0 + 12), title, fill=(20, 31, 43), font=font_title)

    plot = (x0 + 58, y0 + 64, x1 - 24, y1 - 48)
    px0, py0, px1, py1 = plot
    draw.rectangle(plot, outline=(226, 232, 240), width=1)

    all_points = [point for _, values, _ in series for point in values]
    steps = [s for s, _ in all_points]
    values = [v for _, v in all_points]
    min_step, max_step = min(steps), max(steps)
    min_value = min(values) if y_min is None else y_min
    max_value = max(values) if y_max is None else y_max
    if abs(max_value - min_value) < 1e-9:
        max_value += 1.0

    for i in range(5):
        y = py0 + int((py1 - py0) * i / 4)
        draw.line((px0, y, px1, y), fill=(241, 245, 249), width=1)
        value = max_value - (max_value - min_value) * i / 4
        draw.text((x0 + 12, y - 8), f"{value:.2f}", fill=(100, 116, 139), font=small)

    for i in range(5):
        x = px0 + int((px1 - px0) * i / 4)
        draw.line((x, py0, x, py1), fill=(248, 250, 252), width=1)
        step = round(min_step + (max_step - min_step) * i / 4)
        draw.text((x - 8, py1 + 8), str(step), fill=(100, 116, 139), font=small)

    def project(step: int, value: float) -> tuple[int, int]:
        x = px0 + int((step - min_step) / max(max_step - min_step, 1) * (px1 - px0))
        y = py1 - int((value - min_value) / (max_value - min_value) * (py1 - py0))
        return x, y

    legend_x = x0 + 18
    legend_y = y1 - 30
    for name, values_for_series, color in series:
        points = [project(step, value) for step, value in values_for_series]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.rectangle((legend_x, legend_y + 4, legend_x + 16, legend_y + 16), fill=color)
        last_step, last_value = values_for_series[-1]
        draw.text(
            (legend_x + 22, legend_y),
            f"{name} ({last_value:.3f})",
            fill=(20, 31, 43),
            font=font,
        )
        legend_x += 170


def make_tensorboard_chart() -> Path:
    scalars = read_scalars()
    image = Image.new("RGB", (1320, 860), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(17)
    draw.text((40, 28), "TensorBoard Training Curves - TuSimple Run", fill=(20, 31, 43), font=title_font)
    draw.text(
        (40, 68),
        "20 epochs, best checkpoint selected by validation F1 at epoch 18",
        fill=(83, 96, 112),
        font=subtitle_font,
    )

    blue = (37, 99, 235)
    green = (22, 163, 74)
    orange = (234, 88, 12)
    purple = (124, 58, 237)

    draw_panel(
        draw,
        (40, 120, 640, 450),
        "Loss",
        [("train", scalars["train/loss"], blue), ("val", scalars["val/loss"], orange)],
    )
    draw_panel(
        draw,
        (680, 120, 1280, 450),
        "F1 Score",
        [("train", scalars["train/f1"], blue), ("val", scalars["val/f1"], green)],
        y_min=0.74,
        y_max=1.0,
    )
    draw_panel(
        draw,
        (40, 490, 640, 820),
        "Accuracy",
        [("train", scalars["train/accuracy"], blue), ("val", scalars["val/accuracy"], green)],
        y_min=0.82,
        y_max=1.0,
    )
    draw_panel(
        draw,
        (680, 490, 1280, 820),
        "Validation Precision / Recall",
        [("precision", scalars["val/precision"], purple), ("recall", scalars["val/recall"], orange)],
        y_min=0.60,
        y_max=1.0,
    )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    output = ASSET_DIR / "tensorboard_tusimple_curves.png"
    image.save(output)
    return output


def preprocess(path: Path) -> np.ndarray:
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((160, 96), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def predict(path: Path, session: ort.InferenceSession) -> tuple[str, float, float]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: preprocess(path)})[0]
    logit = float(np.asarray(output).reshape(-1)[0])
    probability = sigmoid(logit)
    klass = "in_lane" if probability >= 0.5 else "out_lane"
    return klass, probability, logit


def make_onnx_examples() -> Path:
    samples = [
        (
            "Correct in_lane",
            ROOT / "data/tusimple_binary/test/in_lane/clips_0601_1494452385593783358_20_center.png",
            "in_lane",
        ),
        (
            "Correct out_lane",
            ROOT / "data/tusimple_binary/test/out_lane/clips_0601_1494452579506899721_20_right.png",
            "out_lane",
        ),
        (
            "FP example",
            ROOT / "data/tusimple_binary/test/out_lane/clips_0601_1494452385593783358_20_right.png",
            "out_lane",
        ),
    ]
    session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])

    card_w, card_h = 360, 300
    image = Image.new("RGB", (1180, 430), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    label_font = load_font(18, bold=True)
    font = load_font(15)
    small = load_font(13)
    draw.text((32, 24), "ONNX Runtime Inference Examples - TuSimple Test", fill=(20, 31, 43), font=title_font)
    draw.text(
        (32, 60),
        "Provider: CPUExecutionProvider, threshold: probability_in_lane >= 0.5",
        fill=(83, 96, 112),
        font=small,
    )

    for idx, (title, path, truth) in enumerate(samples):
        x = 32 + idx * (card_w + 24)
        y = 100
        pred, prob, logit = predict(path, session)
        ok = pred == truth
        border = (22, 163, 74) if ok else (234, 88, 12)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=14, fill=(255, 255, 255), outline=border, width=3)
        draw.text((x + 16, y + 14), title, fill=(20, 31, 43), font=label_font)
        roi = Image.open(path).convert("RGB").resize((320, 192), Image.BILINEAR)
        image.paste(roi, (x + 20, y + 52))
        draw.text((x + 20, y + 255), f"truth: {truth}", fill=(83, 96, 112), font=font)
        draw.text((x + 20, y + 278), f"pred: {pred}  p(in)={prob:.4f}", fill=border, font=font)
        draw.text((x + 20, y + 301), f"logit={logit:.4f}", fill=(83, 96, 112), font=small)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    output = ASSET_DIR / "onnx_runtime_tusimple_examples.png"
    image.save(output)
    return output


def main() -> None:
    chart = make_tensorboard_chart()
    examples = make_onnx_examples()
    print(chart)
    print(examples)


if __name__ == "__main__":
    main()
