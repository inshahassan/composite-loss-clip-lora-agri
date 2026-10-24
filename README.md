# Regularized CLIP-LoRA for low-data agricultural image recognition

Code, split protocol and result files for the paper
"Regularized CLIP-LoRA Adaptation for Low-Data Agricultural Image Recognition".

## Contents
- `agroonto/` - model (CLIP ViT-B/32 + LoRA), taxonomy similarity matrices, composite loss, training loop, data loading
- `run_final_protocol.py` - M1 (composite), M2 (cross-entropy), M3 (label-smoothing control), M4 (linear probe); full data, seeds 42/123/456
- `run_extra_experiments.py` - low-data (10 %, 25 %) runs and controls: shuffled taxonomy, no taxonomy-KL, M3
- `run_lambda_sweep.py` - earlier lambda_onto sweep (it used the original validation split, which is the test split in the paper; see the paper's limitations)
- `run_published_methods.py`, `run_published_final.py`, `run_baselines.py` - simplified re-implementations of published methods used as baselines (not the authors' code)
- `aggregate_final.py`, `aggregate_extra.py`, `make_*.py` - tables and figures
- `results/` - per-run JSON files (accuracy, macro-F1, history, split sizes) behind every table and figure

## Setup
```
pip install -r requirements.txt
```
Download PlantVillage (tomato classes), DeepWeeds and IP102 from their official pages and place them under `data/` as `data/plantvillage/PlantVillage`, `data/deepweeds/{images,labels}` and `data/ip102/IP102_YOLOv5/{images,labels}` (paths are set in `agroonto/config.py`).

## Protocol
Test = original validation split of each dataset. Validation = 15 % stratified hold-out of the training set (split seed 2024); the best epoch is chosen on validation only. 15 epochs, three seeds. Low-data subsets: 10 % and 25 % stratified (seed 2024).

## Datasets
- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- DeepWeeds: https://www.tensorflow.org/datasets/catalog/deep_weeds
- IP102: https://mmcheng.net/ip102/
