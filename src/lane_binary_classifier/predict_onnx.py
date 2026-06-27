from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from lane_binary_classifier.data import ImageConfig, TARGET_TO_CLASS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime inference on one ROI image.")
    parser.add_argument("--onnx-model", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--image", required=True, help="Path to an RGB ROI image.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime provider, for example CPUExecutionProvider or CUDAExecutionProvider.",
    )
    return parser.parse_args()


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def preprocess(image_path: str | Path, image_config: ImageConfig) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_config.width, image_config.height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(image_config.mean, dtype=np.float32)) / np.array(
        image_config.std,
        dtype=np.float32,
    )
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def main() -> None:
    args = parse_args()
    image_config = ImageConfig(width=args.width, height=args.height)

    available = ort.get_available_providers()
    provider = args.provider if args.provider in available else "CPUExecutionProvider"
    session = ort.InferenceSession(args.onnx_model, providers=[provider])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    image = preprocess(args.image, image_config)
    output = session.run([output_name], {input_name: image})[0]

    logit = float(np.asarray(output).reshape(-1)[0])
    probability = float(sigmoid(np.array([logit], dtype=np.float32))[0])
    target = 1 if probability >= args.threshold else 0

    print(
        f"class={TARGET_TO_CLASS[target]} "
        f"probability_in_lane={probability:.4f} "
        f"logit={logit:.4f} "
        f"provider={provider}"
    )


if __name__ == "__main__":
    main()
