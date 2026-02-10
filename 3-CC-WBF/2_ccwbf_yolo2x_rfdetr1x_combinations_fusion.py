#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two YOLO models + RF-DETR fusion using Contributor Consensus Weighted Box Fusion (CC-WBF)
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from rfdetr import RFDETRBase

# -----------------------------
# CC-WBF functions
# -----------------------------
def modified_weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                   weights=None, iou_thr=0.55,
                                   conf_mode="weighted", rescale_mode="min"):
    num_models = len(boxes_list)
    if weights is None:
        weights = np.ones(num_models)
    else:
        weights = np.array(weights, dtype=float)

    fused_rows = []
    for label in set([l for sub in labels_list for l in sub]):
        # collect all boxes for this label
        arrs = []
        for t, (b, s, l) in enumerate(zip(boxes_list, scores_list, labels_list)):
            for bi, si, li in zip(b, s, l):
                if li == label:
                    arrs.append([label, si*weights[t], t, *bi])
        if not arrs:
            continue
        arr = np.array(arrs)
        # simple CC-WBF: weighted average by confidence
        Ci = arr[:, 1]
        boxes = arr[:, 3:7]
        fused_xyxy = np.sum(boxes * Ci[:, None], axis=0) / np.sum(Ci)
        if conf_mode == "weighted":
            conf = np.sum(Ci**2)/np.sum(Ci)
        elif conf_mode == "avg":
            conf = np.mean(Ci)
        else:
            conf = np.max(Ci)
        # rescale by number of unique models
        M = len(np.unique(arr[:, 2]))
        if rescale_mode == "min":
            conf *= min(M, num_models)/num_models
        elif rescale_mode == "linear":
            conf *= M/num_models
        # store
        fused_rows.append([label, conf, *fused_xyxy])
    if fused_rows:
        fused = np.array(fused_rows)
        out_boxes = fused[:, 2:6]
        out_scores = fused[:, 1]
        out_labels = fused[:, 0].astype(int)
    else:
        out_boxes = np.zeros((0,4))
        out_scores = np.zeros((0,))
        out_labels = np.zeros((0,), dtype=int)
    return out_boxes, out_scores, out_labels

# -----------------------------
# Dataset helpers
# -----------------------------
def get_image_wh(img_path):
    with Image.open(img_path) as im:
        return im.size  # w,h

def load_yolo_gt(label_path):
    boxes, labels = [], []
    if not Path(label_path).exists():
        return boxes, labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            x1 = max(0, xc - w/2)
            y1 = max(0, yc - h/2)
            x2 = min(1, xc + w/2)
            y2 = min(1, yc + h/2)
            boxes.append([x1, y1, x2, y2])
            labels.append(cid)
    return boxes, labels

# -----------------------------
# YOLO prediction
# -----------------------------
@torch.no_grad()
def yolo_predict_normxyxy(yolo_model, img_path, w, h, conf_th, iou_th, imgsz, device):
    res = yolo_model.predict(source=img_path, conf=conf_th, iou=iou_th, imgsz=imgsz,
                             device=0 if device.startswith("cuda") else device, verbose=False)
    if not res or len(res[0].boxes) == 0:
        return [], [], []
    r = res[0]
    b = r.boxes.xyxy.cpu().numpy()
    s = r.boxes.conf.cpu().numpy()
    c = r.boxes.cls.cpu().numpy().astype(int)
    b[:, [0,2]] /= w
    b[:, [1,3]] /= h
    return b.tolist(), s.tolist(), c.tolist()

# -----------------------------
# RF-DETR prediction
# -----------------------------
@torch.no_grad()
def rfdetr_predict_normxyxy(rf_model, img_path, w, h, device):
    preds = rf_model.predict(str(img_path), threshold=0.0)
    if not preds:
        return [], [], []
    pred = preds[0]
    boxes = np.array(pred.xyxy)
    scores = np.array(pred.confidence)
    labels = np.array(pred.class_id)
    boxes[:, [0,2]] /= w
    boxes[:, [1,3]] /= h
    return boxes.tolist(), scores.tolist(), labels.tolist()

# -----------------------------
# Main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", required=True, choices=["train","valid","test"])
    parser.add_argument("--yolo_models", nargs="+", required=True)
    parser.add_argument("--rfdetr_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--low_conf", type=float, default=0.001)
    parser.add_argument("--final_conf", type=float, default=0.25)
    parser.add_argument("--wbf_iou", type=float, default=0.55)
    parser.add_argument("--conf_mode", choices=["weighted","avg","max"], default="weighted")
    parser.add_argument("--rescale_mode", choices=["min","linear","none"], default="min")
    parser.add_argument("--weights", type=str, default="1.0,1.0,1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_per_image", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_weights = [float(w) for w in args.weights.split(",")]

    # load RF-DETR
    rf_model = RFDETRBase(num_classes=2, pretrained=False)
    ckpt = torch.load(args.rfdetr_ckpt, map_location=args.device)
    rf_model.model.model.load_state_dict(ckpt, strict=False)
    rf_model.model.model.to(args.device)
    rf_model.model.model.eval()

    # load YOLO models
    yolo_models = [YOLO(p) for p in args.yolo_models]

    # images
    img_dir = Path(args.dataset_root) / args.split
    img_paths = list(img_dir.glob("*.*"))

    for img_path in tqdm(img_paths):
        w,h = get_image_wh(img_path)
        boxes_list, scores_list, labels_list = [], [], []

        # all YOLO pairs
        for i in range(len(yolo_models)):
            for j in range(i+1, len(yolo_models)):
                for ym in [yolo_models[i], yolo_models[j]]:
                    b,s,l = yolo_predict_normxyxy(ym, img_path, w, h,
                                                  args.low_conf, 0.7, args.imgsz, args.device)
                    boxes_list.append(b)
                    scores_list.append(s)
                    labels_list.append(l)
                # RF-DETR predictions
                b,s,l = rfdetr_predict_normxyxy(rf_model, img_path, w, h, args.device)
                boxes_list.append(b)
                scores_list.append(s)
                labels_list.append(l)

                fused_boxes, fused_scores, fused_labels = modified_weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list,
                    weights=fusion_weights,
                    iou_thr=args.wbf_iou,
                    conf_mode=args.conf_mode,
                    rescale_mode=args.rescale_mode
                )

                # save or print fused results
                print(f"{img_path.name}: {len(fused_boxes)} boxes")

if __name__ == "__main__":
    main()
