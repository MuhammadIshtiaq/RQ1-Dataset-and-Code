import argparse
import os
import re
from datetime import datetime
from ultralytics import YOLO


def sanitize_name(s: str) -> str:
    """Make a safe run-name token from a model path/name."""
    base = os.path.basename(s).replace(".pt", "")
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base[:80]


def parse_args():
    parser = argparse.ArgumentParser("YOLO Multi-Model Training Script")

    # ✅ multiple models
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more YOLO model names/paths (e.g., yolov8s.pt yolov8m.pt yolov9t.pt)"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id (e.g., 0 or 0,1)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience"
    )

    # ⚠️ Ultralytics resume resumes *one run*. With multi-model, it’s safer to default OFF.
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training (use carefully; resumes from the last run checkpoint)"
    )

    # ✅ nice-to-have: put all runs under one folder
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for runs"
    )

    # ✅ nice-to-have: avoid overwriting
    parser.add_argument(
        "--name_prefix",
        type=str,
        default="multi",
        help="Prefix for each run name (model name will be appended)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # timestamp helps keep runs grouped & unique
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_path in args.models:
        run_name = f"{args.name_prefix}_{sanitize_name(model_path)}_{ts}"
        print("\n" + "=" * 70)
        print(f"Training model: {model_path}")
        print(f"Run name      : {run_name}")
        print("=" * 70)

        model = YOLO(model_path)
        model.info()

        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            resume=args.resume,
            project=args.project,
            name=run_name
        )


if __name__ == "__main__":
    main()
