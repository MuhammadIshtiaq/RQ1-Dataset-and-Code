import argparse
import os
import sys
import types
import re
from typing import Any, Optional, Set, Tuple

import torch
import torch.nn as nn

# ----------------------------------------
# PATCH: repair broken typing_extensions for pydantic/pydantic_core/typing_inspection
# ----------------------------------------
import typing_extensions as te

if not hasattr(te, "Sentinel"):
    class Sentinel:
        def __init__(self, name, repr=None, module=None):
            self.name = name
            self.repr = repr if repr is not None else name
            self.module = module

        def __repr__(self):
            return self.repr

    te.Sentinel = Sentinel
    sys.modules["typing_extensions"].Sentinel = Sentinel
    print("[PATCH] Injected typing_extensions.Sentinel")

if not hasattr(te, "deprecated"):
    def deprecated(*args, **kwargs):
        def wrapper(obj):
            return obj
        return wrapper
    te.deprecated = deprecated
    sys.modules["typing_extensions"].deprecated = deprecated
    print("[PATCH] Injected typing_extensions.deprecated")

if not hasattr(te, "LiteralString"):
    te.LiteralString = str
    sys.modules["typing_extensions"].LiteralString = str
    print("[PATCH] Injected typing_extensions.LiteralString")

if not hasattr(te, "TypeAliasType"):
    class _FakeTypeAliasType:
        def __init__(self, *args, **kwargs):
            self.__args__ = args
            self.__kwargs__ = kwargs

        def __repr__(self):
            return "TypeAliasType(fake)"

    te.TypeAliasType = _FakeTypeAliasType
    sys.modules["typing_extensions"].TypeAliasType = _FakeTypeAliasType
    print("[PATCH] Injected typing_extensions.TypeAliasType")

if not hasattr(te, "TypeIs"):
    class _FakeTypeIs:
        def __class_getitem__(cls, item):
            return bool

    te.TypeIs = _FakeTypeIs
    sys.modules["typing_extensions"].TypeIs = _FakeTypeIs
    print("[PATCH] Injected typing_extensions.TypeIs")

class _FakeTypedDictMeta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        return type.__new__(mcls, name, (dict,), dict(namespace))

class _FakeTypedDict(dict, metaclass=_FakeTypedDictMeta):
    pass

te.TypedDict = _FakeTypedDict
sys.modules["typing_extensions"].TypedDict = _FakeTypedDict
print("[PATCH] Injected typing_extensions.TypedDict (closed= compatible)")


# ----------------------------------------
# Now safe to import rfdetr
# ----------------------------------------
from rfdetr import RFDETRBase
import rfdetr.engine as rfd_engine


# ----------------------------
# Dummy roboflow module
# ----------------------------
if "roboflow" not in sys.modules:
    fake_rf = types.ModuleType("roboflow")
    fake_rf.__dict__.update({"__version__": "0.0.0", "Roboflow": object})
    sys.modules["roboflow"] = fake_rf


def disable_amp_in_rfdetr():
    """
    Force RF-DETR to run float32 only (disable autocast/AMP).
    """
    def get_autocast_args_disabled(args):
        return dict(enabled=False, cache_enabled=False)

    rfd_engine.get_autocast_args = get_autocast_args_disabled
    print("[INFO] RF-DETR autocast/AMP disabled: float32 only.")


# ----------------------------
# Checkpoint -> state_dict extractor
# ----------------------------
def _extract_state_dict(ckpt: dict, prefer_ema: bool = True) -> Optional[dict]:
    if not isinstance(ckpt, dict):
        return None

    if prefer_ema:
        for k in ["ema", "ema_state_dict", "model_ema", "state_dict_ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]

    for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]

    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    return None


# ----------------------------
# Robust torch.nn.Module finder (handles hidden properties)
# ----------------------------
def _find_any_torch_module(obj: Any, max_depth: int = 4) -> Optional[nn.Module]:
    """
    RFDETRBase in your build hides the nn.Module behind properties/internal objects.
    This function scans:
      - direct instance check
      - common attribute names
      - ALL attributes via dir(obj) + getattr (safe)
      - recursion into child objects (limited depth)
    """
    seen: Set[int] = set()

    def _walk(x: Any, depth: int) -> Optional[nn.Module]:
        if x is None:
            return None

        xid = id(x)
        if xid in seen:
            return None
        seen.add(xid)

        if isinstance(x, nn.Module):
            return x

        if depth <= 0:
            return None

        # Try common names first
        for name in ["model", "net", "module", "detector", "engine", "trainer", "core", "_model", "_net"]:
            if hasattr(x, name):
                try:
                    v = getattr(x, name)
                except Exception:
                    v = None
                m = _walk(v, depth - 1)
                if m is not None:
                    return m

        # Scan ALL attributes, including properties
        try:
            names = dir(x)
        except Exception:
            names = []

        for name in names:
            if name.startswith("__") and name.endswith("__"):
                continue
            # Skip obvious non-object fields
            if name in {"training", "device", "dtype"}:
                continue

            try:
                v = getattr(x, name)
            except Exception:
                continue

            # Fast path
            if isinstance(v, nn.Module):
                return v

            # Avoid recursing into huge modules like torch, sys
            if isinstance(v, (int, float, str, bytes, bool, type, types.ModuleType)):
                continue

            m = _walk(v, depth - 1)
            if m is not None:
                return m

        return None

    return _walk(obj, max_depth)


def load_stage1_weights(rfdetr_obj: Any, ckpt_path: str, prefer_ema: bool = True) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt, prefer_ema=prefer_ema)
    if state is None:
        raise RuntimeError(
            "Could not find a valid state_dict inside the checkpoint. "
            "Tried EMA keys and common keys."
        )

    # strip DataParallel prefix if any
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "") if k.startswith("module.") else k] = v

    # Find underlying nn.Module (your build hides it)
    torch_model = _find_any_torch_module(rfdetr_obj, max_depth=5)
    if torch_model is None:
        raise RuntimeError(
            "Still cannot find torch.nn.Module inside RFDETRBase.\n"
            "Your RF-DETR build hides it deeper than expected.\n"
            "If this happens, send me: `python -c \"from rfdetr import RFDETRBase; m=RFDETRBase(num_classes=2,pretrained=False); print(type(m)); print([a for a in dir(m) if 'model' in a.lower() or 'net' in a.lower()])\"`"
        )

    missing, unexpected = torch_model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] Loaded Stage-1 weights from: {ckpt_path}")
    print(f"[INFO] load_state_dict(strict=False) done.")
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing first 20)")
        for k in missing[:20]:
            print("  -", k)
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing first 20)")
        for k in unexpected[:20]:
            print("  -", k)


# ======================================================================
# One-terminal epoch progress bar (tqdm) by parsing printed epoch lines
# ======================================================================
class StdoutEpochProgress:
    def __init__(self, real_stdout, total_epochs: int):
        self.real_stdout = real_stdout
        self.total_epochs = max(1, int(total_epochs))
        self._buf = ""
        self._last_epoch = 0

        try:
            from tqdm import tqdm
        except Exception as e:
            raise RuntimeError("Install tqdm: pip install tqdm") from e

        self.tqdm = tqdm(total=self.total_epochs, unit="epoch", dynamic_ncols=True, leave=True)
        self.tqdm.set_description("Training")

        self._patterns = [
            re.compile(r"(?:^|\s)Epoch\s*[:#]?\s*(\d+)", re.IGNORECASE),
            re.compile(r"Epoch\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE),
            re.compile(r"(\d+)\s*/\s*(\d+)\s*epochs", re.IGNORECASE),
        ]

    def _maybe_update(self, line: str):
        epoch = None
        total = None

        for pat in self._patterns:
            m = pat.search(line)
            if not m:
                continue
            if len(m.groups()) == 1:
                epoch = int(m.group(1))
            else:
                epoch = int(m.group(1))
                total = int(m.group(2))
            break

        if total is not None and total > 0 and total != self.total_epochs:
            self.total_epochs = total
            self.tqdm.total = total
            self.tqdm.refresh()

        if epoch is not None and epoch > self._last_epoch:
            inc = epoch - self._last_epoch
            self._last_epoch = epoch
            self.tqdm.update(inc)

    def write(self, s: str):
        self.real_stdout.write(s)
        self.real_stdout.flush()

        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._maybe_update(line)

    def flush(self):
        self.real_stdout.flush()

    def close(self):
        try:
            if self._last_epoch < self.total_epochs:
                self.tqdm.update(self.total_epochs - self._last_epoch)
        except Exception:
            pass
        self.tqdm.close()


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("RF-DETR Stage-2 fine-tune (loads Stage-1 EMA/base weights)")

    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--prefer_ema", action="store_true")

    p.add_argument("--num_classes", type=int, default=2)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=2)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_encoder", type=float, default=5e-6)

    p.add_argument("--imgsz", type=int, default=672)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", type=str, default="rfdetr-weapons")
    p.add_argument("--run", type=str, default="stage2")

    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0005)
    p.add_argument("--no_early_stop", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.isdir(args.data):
        raise FileNotFoundError(f"Dataset dir not found: {args.data}")

    os.makedirs(args.out, exist_ok=True)
    print(f"Output directory: {args.out}")

    print(f"[INFO] Initializing RF-DETR (pretrained=False) with classes: {args.num_classes}")
    model = RFDETRBase(num_classes=args.num_classes, pretrained=False)

    disable_amp_in_rfdetr()

    # Load Stage-1 weights into the hidden torch model
    load_stage1_weights(model, args.stage1_ckpt, prefer_ema=args.prefer_ema)

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
        use_ema=True,
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
        print(f"[INFO] Resuming Stage-2 from: {args.resume}")
        train_kwargs["resume"] = args.resume

    print("=== Stage-2 Training configuration ===")
    for k, v in train_kwargs.items():
        print(f"{k}: {v}")
    print("======================================")

    # One terminal progress bar
    old_stdout = sys.stdout
    pbar_stdout = StdoutEpochProgress(sys.stdout, total_epochs=args.epochs)
    sys.stdout = pbar_stdout

    try:
        model.train(**train_kwargs)
    finally:
        sys.stdout = old_stdout
        pbar_stdout.close()

    print("Stage-2 training finished.")
    print("Saved in:", args.out)
    print("Look for checkpoints:")
    print("  - checkpoint_best_ema.pth")
    print("  - checkpoint_best_total.pth")
    print("  - checkpoint_best_regular.pth")


if __name__ == "__main__":
    main()
