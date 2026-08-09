"""Train the specialized SmartPack canned-vegetable detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, such as 0 for CUDA or cpu. Default: automatic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = PROJECT_DIR / "dataset" / "canned_vegtables" / "data.yaml"
    base_weights = PROJECT_DIR / "yolo26n.pt"
    if not data_file.is_file():
        raise FileNotFoundError(
            f"Missing {data_file}. Generate the dataset with dataset_generator.py first."
        )
    if not base_weights.is_file():
        raise FileNotFoundError(f"Missing base model weights: {base_weights}")

    model = YOLO(str(base_weights))
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        auto_augment=None,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name="cans",
    )


if __name__ == "__main__":
    main()
