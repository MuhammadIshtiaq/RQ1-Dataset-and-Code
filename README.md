# Consensus-Driven Bounding Box Fusion for Robust Weapon Detection in Surveillance Systems
*(Source code and experiment pipeline for the manuscript submitted to **The Journal of Supercomputing**)*

**Status:** Submitted to *The Journal of Supercomputing*  
**Repository:** Private during review; will be made public upon acceptance  
**Zenodo DOI:** _to be inserted after reservation_

This repository contains the full experimental pipeline for reproducibility of the paper:

> **Consensus-Driven Bounding Box Fusion for Robust Weapon Detection in Surveillance Systems**  
> Authors: Muhammad Ishtiaq et al.

---

## Repository Layout

```
1-YOLO Trainings/
├── 1_train_multiple_yolo_models_on_coco_weights.py         # to train multiple YOLO models using COCO weights cumulatively
├── 2_train_multiple_yolo_models_on_custom_weights.py       # to train multiple YOLO models using custom weights cumulatively
└── CLI_run_commands.txt                                    # guide to run the trainings

2-RF-DETR Trainings/
├── 1_train_rfdetr_on_coco_weights.py                       # to train RF-DETR model using COCO weights
├── 2_train_rfdetr_on_custom_weights.py                     # to train RF-DETR model using custom weights
└── CLI_run_commands.txt                                    # guide to run the trainings

3-CC-WBF/
├── 1_ccwbf_yolo_combinations_fusion.py                     # applying CC-WBF on three YOLO models combibations 
├── 2_ccwbf_yolo2x_rfdetr1x_combinations_fusion.py          # applying CC-WBF on a pair of YOLO models and RF-DETR
└── CLI_run_commands.txt                                    # guide to run the trainings

4-Dataset.txt                                               # dataset source and description
USAGE_NOTICE.txt                                            # usage policy and citation instructions
```

---

## Dataset

See **4-Dataset.txt** for detailed notes.

### YOLO Dataset Folder Structure
For example:
```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── valid/
│   │   ├── img_003.jpg
│   │   ├── img_004.jpg
│   │   └── ...
│   └── test/
│       ├── img_005.jpg
│       ├── img_006.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   ├── img_002.txt
│   │   └── ...
│   ├── valid/
│   │   ├── img_003.txt
│   │   ├── img_004.txt
│   │   └── ...
│   └── test/
│       ├── img_005.txt
│       ├── img_006.txt
│       └── ...
```

### RF-DETR Dataset Folder Structure 
For example:
```
rf_detr_dataset/
├── train/
│ ├── img_001.jpg
│ ├── img_002.jpg
│ ├── ...
│ └── _annotations.coco.json
├── valid/
│ ├── img_005.jpg
│ ├── img_006.jpg
│ ├── ...
│ └── _annotations.coco.json
├── test/
│ ├── img_003.jpg
│ ├── img_004.jpg
│ ├── ...
│ └── _annotations.coco.json
```

---

## Environment and Dependencies

This project uses **separate Conda environments** for framework-specific dependencies.  

Install all requirements:

```bash
pip install -r requirements.txt
```

Main packages:
```
# YOLO environment
torch==1.8.1+cu101
torchvision
ultralytics==8.3.156
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
pyyaml

# RF-DETR environment
torch==1.10.1+cu102
torchvision
rfdetr
numpy
pandas
opencv-python
tqdm
Pillow
typing_extensions
```

---

## Reproducing the Experiments

1. **Train YOLO models on COCO weights**  
   - Run `1_train_multiple_yolo_models_on_coco_weights.py` to train all benchmarked YOLO models.

2. **Finetune YOLO models on the custom dataset**  
   - Execute `2_train_multiple_yolo_models_on_custom_weights.py` to finetune the YOLO models using the finetuning dataset described in the paper.

3. **Train RF-DETR on COCO weights**  
   - Run `1_train_rfdetr_on_coco_weights.py` to train the RF-DETR model.

4. **Finetune RF-DETR on the custom dataset**  
   - Execute `2_train_rfdetr_on_custom_weights.py` to finetune the RF-DETR model on the same finetuning dataset.

5. **Apply CC-WBF on YOLO model combinations**  
   - Run `1_ccwbf_yolo_combinations_fusion.py` to perform cross-architecture fusion on YOLO model ensembles.

6. **Apply CC-WBF on YOLO pairs and RF-DETR**  
   - Execute `2_ccwbf_yolo2x_rfdetr1x_combinations_fusion.py` for dual YOLO + RF-DETR fusion experiments.

Each script includes parameters and instructions to reproduce the reported results in the paper.

---

## Citation

Please read `USAGE_NOTICE.txt` for legal terms.  
Once the paper is published, cite it as:

```
@article{AuthorYear,
  title   = {Consensus-Driven Bounding Box Fusion for Robust Weapon Detection in Surveillance Systems},
  author  = {Muhammad Ishtiaq, Mingchu Li and ...},
  journal = {The Journal of Supercomputing},
  year    = {2026},
  doi     = {will be provided after acceptance}
}
```

---

## License & Usage Policy

Copyright © 2025 Muhammad Ishtiaq.

This repository is provided **solely for transparency and peer review**.  
No reuse is permitted without prior written permission. See `USAGE_NOTICE.txt`.

---

## Contact

**First author:** Muhammad Ishtiaq — ishtiaqrai8@gmail.com
