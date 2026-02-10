import argparse
import os
import sys
import types

import torch

# ----------------------------------------
# PATCH: repair broken typing_extensions for pydantic/pydantic_core/typing_inspection
# ----------------------------------------
import typing_extensions as te

# 1) Sentinel (for pydantic_core)
if not hasattr(te, "Sentinel"):
    class Sentinel:
        def __init__(self, name, repr=None, module=None):
            self.name = name
            if repr is None:
                repr = name
            self.repr = repr
            self.module = module

        def __repr__(self):
            return self.repr
    te.Sentinel = Sentinel
    sys.modules["typing_extensions"].Sentinel = Sentinel
    print("[PATCH] Injected typing_extensions.Sentinel to satisfy pydantic_core.")

# 2) deprecated decorator
if not hasattr(te, "deprecated"):
    def deprecated(*args, **kwargs):
        def wrapper(obj):
            return obj
        return wrapper
    te.deprecated = deprecated
    sys.modules["typing_extensions"].deprecated = deprecated
    print("[PATCH] Injected typing_extensions.deprecated to satisfy pydantic_core.")

# 3) LiteralString
if not hasattr(te, "LiteralString"):
    te.LiteralString = str
    sys.modules["typing_extensions"].LiteralString = str
    print("[PATCH] Injected typing_extensions.LiteralString alias to str.")

# 4) TypeAliasType
if not hasattr(te, "TypeAliasType"):
    class _FakeTypeAliasType:
        def __init__(self, *args, **kwargs):
            self.__args__ = args
            self.__kwargs__ = kwargs

        def __repr__(self):
            return "TypeAliasType(fake)"
    te.TypeAliasType = _FakeTypeAliasType
    sys.modules["typing_extensions"].TypeAliasType = _FakeTypeAliasType
    print("[PATCH] Injected typing_extensions.TypeAliasType stub.")

# 5) TypeIs
if not hasattr(te, "TypeIs"):
    class _FakeTypeIs:
        def __class_getitem__(cls, item):
            return bool
    te.TypeIs = _FakeTypeIs
    sys.modules["typing_extensions"].TypeIs = _FakeTypeIs
    print("[PATCH] Injected typing_extensions.TypeIs stub.")

# 6) TypedDict (accepts closed= kwarg)
class _FakeTypedDictMeta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        # ignore kwargs such as closed=True
        return type.__new__(mcls, name, (dict,), dict(namespace))

class _FakeTypedDict(dict, metaclass=_FakeTypedDictMeta):
    pass

te.TypedDict = _FakeTypedDict
sys.modules["typing_extensions"].TypedDict = _FakeTypedDict
print("[PATCH] Injected typing_extensions.TypedDict stub that accepts 'closed='.")


# ----------------------------------------
# Now safe to import rfdetr
# ----------------------------------------
from rfdetr import RFDETRBase
import rfdetr.engine as rfd_engine  # needed so we can patch AMP



# ----------------------------
# Dummy roboflow module
# ----------------------------
if "roboflow" not in sys.modules:
    fake_rf = types.ModuleType("roboflow")
    fake_rf.__dict__.update({
        "__version__": "0.0.0",
        "Roboflow": object,
    })
    sys.modules["roboflow"] = fake_rf


# ----------------------------
# Disable AMP (no bfloat16 on your GPU)
# ----------------------------
def disable_amp_in_rfdetr():
    """
    RF-DETR internally uses:
        with autocast(**get_autocast_args(args)):

    The default implementation can request dtype=bfloat16, which your GPU
    does not support. Here we override get_autocast_args() so that AMP is
    completely disabled and everything runs in float32.
    """
    def get_autocast_args_disabled(args):
        return dict(enabled=False, cache_enabled=False)

    rfd_engine.get_autocast_args = get_autocast_args_disabled
    print("[INFO] RF-DETR autocast/AMP disabled: running in pure float32.")


# ----------------------------
# Argument parser (TRAIN)
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser("RF-DETR training")

    parser.add_argument(
        "--data", type=str, required=True,
        help="Dataset root containing train/ valid/ test/ in COCO format"
    )
    parser.add_argument(
        "--out", type=str, default="runs/rfdetr/exp",
        help="Output directory for checkpoints, logs, etc."
    )

    # ▶ Longer training – DETR-style models need this
    parser.add_argument(
        "--epochs", type=int, default=120,
        help="Number of training epochs (recommend 100–150 for RF-DETR)"
    )

    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)

    # ▶ LR – keep defaults but expose via CLI
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)

    # ▶ Resolution – affects small/medium object AP
    parser.add_argument(
        "--imgsz", type=int, default=896,
        help="Training resolution (try 896 / 960 / 1024)"
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Logging
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="rfdetr-weapons")
    parser.add_argument("--run", type=str, default="exp")
    parser.add_argument("--resume", type=str, default=None)

    # ▶ Early stopping controls (to avoid stopping too early)
    parser.add_argument(
        "--early_stop_patience", type=int, default=20,
        help="Epochs to wait with no improvement before stopping"
    )
    parser.add_argument(
        "--early_stop_min_delta", type=float, default=0.0005,
        help="Minimum metric improvement to reset early-stopping counter"
    )
    parser.add_argument(
        "--no_early_stop", action="store_true",
        help="Disable early stopping entirely"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.isdir(args.data):
        raise FileNotFoundError(f"Dataset directory not found: {args.data}")

    os.makedirs(args.out, exist_ok=True)
    print(f"Output directory: {args.out}")

    # ----------------------------
    # Model (COCO pretrain, 2 classes)
    # ----------------------------
    model = RFDETRBase(num_classes=2, pretrained=True)
    print("[INFO] RF-DETR initialized with COCO pretrained weights + 2-class head")

    # Disable AMP (bfloat16) -> run in float32
    disable_amp_in_rfdetr()

    # ----------------------------
    # Training config
    # ----------------------------
    use_early_stopping = not args.no_early_stop

    train_kwargs = dict(
        dataset_dir=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        resolution=args.imgsz,
        weight_decay=args.weight_decay,
        device=device,
        use_ema=True,                      # EMA is good for final AP
        early_stopping=use_early_stopping,
        early_stopping_patience=args.early_stop_patience,
        early_stopping_min_delta=args.early_stop_min_delta,
        checkpoint_interval=5,
    )

    if args.tensorboard:
        train_kwargs["tensorboard"] = True

    if args.wandb:
        train_kwargs["wandb"] = True
        train_kwargs["project"] = args.project
        train_kwargs["run"] = args.run

    if args.resume:
        print(f"[INFO] Resuming training from checkpoint: {args.resume}")
        train_kwargs["resume"] = args.resume

    # ----------------------------
    # Train model
    # ----------------------------
    print("=== Training configuration ===")
    for k, v in train_kwargs.items():
        print(f"{k}: {v}")
    print("==============================")

    model.train(**train_kwargs)

    print("Training finished.")
    print("Checkpoints and logs saved in:", args.out)
    print("Key files to look for (created by RF-DETR):")
    print("  - checkpoint_best_ema.pth (best EMA weights for inference)")
    print("  - checkpoint_best_total.pth")
    print("  - checkpoint_best_regular.pth")


if __name__ == "__main__":
    main()
