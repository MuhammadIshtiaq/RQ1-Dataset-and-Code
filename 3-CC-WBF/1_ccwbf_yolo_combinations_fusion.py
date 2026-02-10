import os
import argparse
from itertools import combinations
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import yaml

# ==========================================================
# Contributor Consensus Weighted Box Fusion (CC-WBF)
# 1) Transitive clustering (connected components on IoU>tau)
# 2) Weighted confidence: C = sum(C_i^2) / sum(C_i)
# 3) Rescale by M = #unique models contributing (not T=#boxes)
# Boxes expected in normalized xyxy [0,1]
# ==========================================================

def prefilter_boxes(boxes, scores, labels, weights, thr):
    """
    Convert per-model predictions into per-class arrays.
    Each record:
      b = [label, Ci, model_weight, model_id, x1, y1, x2, y2]
    where Ci = score * model_weight
    """
    new_boxes = dict()
    for m in range(len(boxes)):
        if len(boxes[m]) != len(scores[m]) or len(boxes[m]) != len(labels[m]):
            continue

        for j in range(len(boxes[m])):
            s = float(scores[m][j])
            if s < thr:
                continue

            label = int(labels[m][j])
            x1, y1, x2, y2 = boxes[m][j]

            # fix ordering
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            # clip to [0,1]
            x1 = float(np.clip(x1, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            y2 = float(np.clip(y2, 0.0, 1.0))

            # ignore degenerate boxes
            if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
                continue

            Ci = s * float(weights[m])
            b = [float(label), float(Ci), float(weights[m]), float(m), x1, y1, x2, y2]
            new_boxes.setdefault(label, []).append(b)

    # sort each class list by confidence descending
    for k in list(new_boxes.keys()):
        arr = np.array(new_boxes[k], dtype=np.float32)
        if arr.shape[0] == 0:
            del new_boxes[k]
            continue
        new_boxes[k] = arr[arr[:, 1].argsort()[::-1]]
    return new_boxes


def bb_iou_matrix_xyxy(boxes_xyxy):
    """
    Compute full IoU matrix for boxes (N,4) in xyxy.
    Returns (N,N) matrix.
    """
    if boxes_xyxy.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    x1 = boxes_xyxy[:, 0][:, None]
    y1 = boxes_xyxy[:, 1][:, None]
    x2 = boxes_xyxy[:, 2][:, None]
    y2 = boxes_xyxy[:, 3][:, None]

    xx1 = np.maximum(x1, x1.T)
    yy1 = np.maximum(y1, y1.T)
    xx2 = np.minimum(x2, x2.T)
    yy2 = np.minimum(y2, y2.T)

    inter_w = np.maximum(xx2 - xx1, 0.0)
    inter_h = np.maximum(yy2 - yy1, 0.0)
    inter = inter_w * inter_h

    area = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    union = area + area.T - inter
    return inter / np.maximum(union, 1e-10)


def connected_components_from_adjacency(adj):
    """
    adj: (N,N) boolean adjacency matrix
    Return list of components as list of index lists.
    """
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            neigh = np.where(adj[u])[0]
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


def fuse_cluster_mwbf(cluster_boxes, num_models, conf_mode="weighted", rescale_mode="min"):
    """
    cluster_boxes: array (K,9) columns:
      [label, Ci, model_weight, model_id, x1,y1,x2,y2]
    num_models: N
    Returns fused row: [label, C_final, M, x1,y1,x2,y2]
    """

    Ci = cluster_boxes[:, 1].astype(np.float32)         # already includes model weights
    coords = cluster_boxes[:, 4:8].astype(np.float32)
    model_ids = cluster_boxes[:, 3].astype(np.int32)

    # coordinate fusion: confidence-weighted average
    wsum = float(np.sum(Ci))
    if wsum <= 0.0:
        # fallback: simple mean coords
        fused_xyxy = np.mean(coords, axis=0)
    else:
        fused_xyxy = np.sum(coords * Ci[:, None], axis=0) / wsum

    # confidence fusion (your requested improvement):
    # C_weighted = sum(Ci^2)/sum(Ci)
    if conf_mode == "weighted":
        C_raw = float(np.sum(Ci * Ci) / max(wsum, 1e-10))
    elif conf_mode == "avg":
        C_raw = float(np.mean(Ci))
    elif conf_mode == "max":
        C_raw = float(np.max(Ci))
    else:
        C_raw = float(np.sum(Ci * Ci) / max(wsum, 1e-10))

    # M = # unique contributing models (your requested fix)
    M = int(len(np.unique(model_ids)))

    # rescale using M (not T)
    if rescale_mode == "none":
        C_final = C_raw
    elif rescale_mode == "linear":
        # C <- C * (M/N)
        C_final = C_raw * (float(M) / float(num_models))
    else:
        # default: min form, C <- C * min(M,N)/N
        C_final = C_raw * (min(float(M), float(num_models)) / float(num_models))

    label = float(cluster_boxes[0, 0])
    return np.array([label, float(C_final), float(M),
                     fused_xyxy[0], fused_xyxy[1], fused_xyxy[2], fused_xyxy[3]], dtype=np.float32)


def modified_weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_mode="weighted",     # weighted / avg / max
    rescale_mode="min"        # min / linear / none
):
    """
    CC-WBF per class:
    - build graph edges with IoU > iou_thr (transitive closure via connected components)
    - fuse each component using:
        * coords: confidence-weighted average
        * conf: sum(Ci^2)/sum(Ci)
        * rescale: using M unique model contributors
    """

    num_models = len(boxes_list)
    if weights is None:
        weights = np.ones(num_models, dtype=np.float32)
    else:
        weights = np.array(weights, dtype=np.float32)
        if len(weights) != num_models:
            weights = np.ones(num_models, dtype=np.float32)

    filtered = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64)
        )

    fused_all = []

    for label, arr in filtered.items():
        # arr shape: (N,9): [label, Ci, weight, model_id, x1,y1,x2,y2]
        coords = arr[:, 4:8]
        iou_mat = bb_iou_matrix_xyxy(coords)

        # adjacency: IoU > thr AND include self edges
        adj = (iou_mat > float(iou_thr))
        np.fill_diagonal(adj, True)

        comps = connected_components_from_adjacency(adj)

        for comp in comps:
            cluster = arr[np.array(comp, dtype=np.int32)]
            fused_row = fuse_cluster_mwbf(
                cluster_boxes=cluster,
                num_models=num_models,
                conf_mode=conf_mode,
                rescale_mode=rescale_mode
            )
            fused_all.append(fused_row)

    fused_all = np.array(fused_all, dtype=np.float32)
    # sort by confidence desc (column 1)
    fused_all = fused_all[fused_all[:, 1].argsort()[::-1]]

    fused_boxes = fused_all[:, 3:7].astype(np.float32)
    fused_scores = fused_all[:, 1].astype(np.float32)
    fused_labels = fused_all[:, 0].astype(np.int64)
    return fused_boxes, fused_scores, fused_labels


# =========================
# Evaluation helpers
# =========================
def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    union = area_a + area_b - inter
    return inter / max(union, 1e-10)

def evaluate_predictions(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, iou_thr=0.5):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    preds = list(zip(pred_boxes, pred_labels, pred_scores))
    preds.sort(key=lambda x: x[2], reverse=True)

    matched = [False] * len(gt_boxes)
    TP = FP = 0

    for p_box, p_lab, _ in preds:
        best_iou = 0.0
        best_idx = -1
        for i, (g_box, g_lab) in enumerate(zip(gt_boxes, gt_labels)):
            if matched[i]:
                continue
            if int(p_lab) != int(g_lab):
                continue
            iou = calculate_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx != -1 and best_iou >= iou_thr:
            matched[best_idx] = True
            TP += 1
        else:
            FP += 1

    FN = sum(1 for m in matched if not m)
    return TP, FP, FN

def prf1(TP, FP, FN):
    p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def read_yolo_label_file(label_path):
    gt_boxes, gt_labels = [], []
    if not os.path.exists(label_path):
        return gt_boxes, gt_labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc = float(parts[1]); yc = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
            x1 = max(0.0, xc - bw / 2)
            y1 = max(0.0, yc - bh / 2)
            x2 = min(1.0, xc + bw / 2)
            y2 = min(1.0, yc + bh / 2)
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(cls)
    return gt_boxes, gt_labels


# =========================
# Data.yaml parsing helpers
# =========================
def _resolve_path(root, p):
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(root, p))

def load_split_dirs_from_data_yaml(data_yaml_path, split):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    root = data.get("path", "")
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    if root and not os.path.isabs(root):
        root = os.path.normpath(os.path.join(yaml_dir, root))
    elif not root:
        root = yaml_dir

    key = "val" if split == "valid" else "test"
    images_dir = data.get(key, None)
    if images_dir is None:
        raise KeyError(f"'{key}' not found in data yaml: {data_yaml_path}")

    images_dir = _resolve_path(root, images_dir)
    labels_dir = images_dir.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    return images_dir, labels_dir


# =========================
# Model discovery
# =========================
def discover_models(models_dir):
    all_pts = []
    for fn in os.listdir(models_dir):
        if fn.lower().endswith(".pt"):
            all_pts.append(os.path.join(models_dir, fn))
    all_pts.sort()
    return all_pts

def nice_model_name(pt_path):
    return os.path.splitext(os.path.basename(pt_path))[0]


# =========================
# Main
# =========================
def parse_args():
    ap = argparse.ArgumentParser("CC-WBF triplet evaluation (transitive+weightedconf+M-rescale)")
    ap.add_argument("--split", type=str, required=True, choices=["valid", "test"])
    ap.add_argument("--models_dir", type=str, required=True)
    ap.add_argument("--data_yaml", type=str, required=True)

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--low_conf", type=float, default=0.001, help="candidate threshold before CC-WBF")
    ap.add_argument("--final_conf", type=float, default=0.25, help="final threshold after CC-WBF")
    ap.add_argument("--mwbf_iou", type=float, default=0.55, help="IoU threshold for graph edges")
    ap.add_argument("--eval_iou", type=float, required=True, help="IoU threshold for TP/FP/FN")

    ap.add_argument("--conf_mode", type=str, default="weighted", choices=["weighted", "avg", "max"],
                    help="Confidence fusion: weighted uses sum(C^2)/sum(C)")
    ap.add_argument("--rescale_mode", type=str, default="min", choices=["min", "linear", "none"],
                    help="Rescale using M unique models: min=>min(M,N)/N, linear=>M/N")

    ap.add_argument("--out_dir", type=str, default=".")
    return ap.parse_args()

def main():
    args = parse_args()

    models_dir = os.path.abspath(args.models_dir)
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    models = discover_models(models_dir)
    if len(models) < 3:
        raise ValueError(f"Need at least 3 .pt models in {models_dir}, found {len(models)}")

    model_names = [nice_model_name(p) for p in models]
    triplets = list(combinations(list(range(len(models))), 3))
    print(f"Discovered {len(models)} models -> {len(triplets)} unique triplets")

    images_dir, labels_dir = load_split_dirs_from_data_yaml(args.data_yaml, args.split)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir for split='{args.split}' not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels dir for split='{args.split}' not found: {labels_dir}")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    image_files.sort()
    print(f"Split={args.split} | Images={len(image_files)}")

    results_rows = []

    for (i, j, k) in triplets:
        tri_name = f"{model_names[i]}+{model_names[j]}+{model_names[k]}"
        print(f"\n=== Triplet: {tri_name} ===")

        y1 = YOLO(models[i])
        y2 = YOLO(models[j])
        y3 = YOLO(models[k])
        yolo_models = [y1, y2, y3]

        totals = {"TP": 0, "FP": 0, "FN": 0}

        for img_file in tqdm(image_files, desc=tri_name, leave=False):
            img_path = os.path.join(images_dir, img_file)
            base = os.path.splitext(img_file)[0]
            label_path = os.path.join(labels_dir, base + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            gt_boxes, gt_labels = read_yolo_label_file(label_path)

            boxes_list, scores_list, labels_list = [], [], []

            for m in yolo_models:
                pred = m.predict(
                    img_path,
                    imgsz=args.imgsz,
                    conf=args.low_conf,
                    device=args.device,
                    verbose=False
                )

                boxes, scores, labels = [], [], []
                for r in pred:
                    for b in r.boxes:
                        x1p, y1p, x2p, y2p = b.xyxy.cpu().numpy()[0]
                        boxes.append([x1p / w, y1p / h, x2p / w, y2p / h])
                        scores.append(float(b.conf.item()))
                        labels.append(int(b.cls.item()))
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)

            fused_boxes, fused_scores, fused_labels = modified_weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=None,
                iou_thr=args.mwbf_iou,
                skip_box_thr=args.low_conf,
                conf_mode=args.conf_mode,
                rescale_mode=args.rescale_mode
            )

            if len(fused_scores) > 0:
                keep = fused_scores >= args.final_conf
                fused_boxes = fused_boxes[keep]
                fused_scores = fused_scores[keep]
                fused_labels = fused_labels[keep]

            TP, FP, FN = evaluate_predictions(
                gt_boxes, gt_labels,
                fused_boxes, fused_scores, fused_labels,
                iou_thr=args.eval_iou
            )

            totals["TP"] += TP
            totals["FP"] += FP
            totals["FN"] += FN

        P, R, F1 = prf1(totals["TP"], totals["FP"], totals["FN"])
        results_rows.append({
            "Triplet": tri_name,
            "Model_1": model_names[i],
            "Model_2": model_names[j],
            "Model_3": model_names[k],
            "TP": totals["TP"],
            "FP": totals["FP"],
            "FN": totals["FN"],
            "Precision": round(P, 6),
            "Recall": round(R, 6),
            "F1": round(F1, 6),
            "split": args.split,
            "eval_iou": args.eval_iou,
            "mwbf_iou": args.mwbf_iou,
            "low_conf": args.low_conf,
            "final_conf": args.final_conf,
            "imgsz": args.imgsz,
            "device": args.device,
            "conf_mode": args.conf_mode,
            "rescale_mode": args.rescale_mode
        })

        print(f"TP={totals['TP']} FP={totals['FP']} FN={totals['FN']} | P={P:.4f} R={R:.4f} F1={F1:.4f}")

    df = pd.DataFrame(results_rows)
    eval_iou_str = f"{args.eval_iou:.2f}".replace(".", "p")
    out_name = f"mwbf_triplets_{args.split}_evalIoU_{eval_iou_str}.xlsx"
    out_path = os.path.join(os.path.abspath(args.out_dir), out_name)

    df.to_excel(out_path, index=False)
    print(f"\n✅ Saved Excel: {out_path}")

    top = df.sort_values("F1", ascending=False).head(10)
    print("\nTop-10 triplets by F1:")
    print(top[["Triplet", "Precision", "Recall", "F1", "TP", "FP", "FN"]].to_string(index=False))

if __name__ == "__main__":
    main()
