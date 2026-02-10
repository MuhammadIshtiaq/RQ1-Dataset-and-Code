import argparse
import os
import re
from datetime import datetime
from ultralytics import YOLO


def sanitize_name(s: str) -> str:
    base = os.path.basename(s).replace(".pt", "")
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base[:80]


def parse_args():
    parser = argparse.ArgumentParser("YOLO Multi-Model Training from Custom Weights")

    # model identifiers (used for naming/logging)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model identifiers (e.g., yolov8s yolov8l yolov9t yolo11s)"
    )

    # directory that contains pretrained .pt files
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Directory containing pretrained model weights (*.pt)"
    )

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name_prefix", type=str, default="exp")

    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_id in args.models:
        weight_file = os.path.join(args.weights_dir, f"{model_id}_best.pt")

        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"❌ Weights not found: {weight_file}")

        run_name = f"{args.name_prefix}_{sanitize_name(model_id)}_from_coco_{ts}"

        print("\n" + "=" * 80)
        print(f"Model ID     : {model_id}")
        print(f"Using weights: {weight_file}")
        print(f"Run name     : {run_name}")
        print("=" * 80)

        model = YOLO(weight_file)   # ✅ load pretrained weights directly
        model.info()

        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            project=args.project,
            name=run_name
        )


if __name__ == "__main__":
    main()
