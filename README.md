# Lite-TacFormer

Tactile material recognition with a lightweight Transformer: a **source-domain classifier** trained on multiple operators plus an **ultra-fast few-shot personal calibration** protocol (BN statistics adaptation + classification head + LoRA adapters) that adapts a pretrained backbone to a new operator using only **20 samples (5 per material)**, trained **on CPU in ~4 seconds**.

Companion code of the ICASSP 2027 submission. The dataset is **not** distributed in this repository.

## Sensor & Data

- **Sensor**: PaXini PX6AX-GEN3-CP-L5325-Omega tactile sensing array — 239 taxels × 3-axis (normal force `fz` and shear forces `fx`/`fy`), USB interface.
- **Sample**: a 150-step time series of 3 force channels, whitened per channel with global train-set statistics → shape `[150, 3]`.
- **Materials** (4 classes): `iron`, `paper`, `plastic`, `pu`.
- **Subjects**: source-domain train operators `OY` / `LG` / `YZ` / `ZSY`; target operators `CDQ` / `CWB` / `GQH` (different collection protocol, never seen in training).

## Repository Layout

```
exp_class.py                          # BaselineTactileTransformer (shared architecture)
classification/
  train_exp_5_compared.py             # 4-person classifier: 6 input-representation comparison
  best_4person.pt                     # trained on OY+LG+YZ+ZSY (4069 samples), peaknorm_nonorm repr
few_shot/
  train_few_shot.py                   # v1: freeze backbone, tune head only (head_only baseline)
  train_few_shot_upgrade.py           # v2 (final method): BN stats adaptation + head + LoRA(r=8)
  ckpt/
    exp3_best_CDQ.pt                  # official CDQ-calibrated weights (paper seed 42)
    exp3_lora_head_CDQ_seed42.json    # official CDQ result log (63.57 %)
checkpoints/
  backbone_3person_OY_LG_YZ.pt        # source backbone trained on OY+LG+YZ, used by few-shot scripts
```

### Model architecture (`exp_class.py`)

Conv1d token projection (3 → 64 channels, kernel 5) + BatchNorm + ReLU; learnable positional
embedding (150 × 64); 2-layer / 4-head Transformer encoder (d=64, FFN 256); global average
pooling; MLP head (64 → 32 → ReLU → Dropout → 4 classes).

## 1. Source-domain classifier (4-person training → zero-shot)

```bash
python classification/train_exp_5_compared.py --repr peaknorm_nonorm   # or --repr all
```

6 input representations are compared (`peaknorm_norm`, `peaknorm_nonorm`, `fft`, `diff`,
`friction`, `friction_phys`). Protocol: 50 epochs, batch 32, AdamW lr 1e-3 wd 1e-4,
CosineAnnealing, final-epoch weights (no validation-set selection). Running
`peaknorm_nonorm` with the released `best_4person.pt` reproduces the paper's cross-person
zero-shot numbers: **CDQ 36.94 % / CWB 25.17 % / GQH 40.14 % / merged 34.06 %**.

## 2. Few-shot personal calibration (final method)

For a new operator: collect **20 samples (5 per material)**, then on CPU:

```bash
python few_shot/train_few_shot_upgrade.py --method lora_head --test-person CDQ \
    --calib-epochs 200 --seed 42
```

Pipeline:

1. Load source backbone (OY+LG+YZ), freeze all parameters.
2. **BN statistics adaptation**: 10 no-gradient train-mode forward passes on the 20 support
   samples — realigns running stats to the new operator's force scale.
3. **Head + LoRA fine-tuning**: LoRA (r=8, α=8, scale=α/r) mounted on the FFN `linear1` /
   `linear2` of each Transformer layer; 200 epochs, AdamW lr 5e-3, on the 20 samples only.
4. Blind evaluation on the operator's remaining data (query set).

Reproduces the paper's seed-42 CDQ result: query accuracy **0.2586 → 0.6357 (+37.7 pt)**,
per-class iron 0.823 / paper 0.846 / plastic 0.600 / pu 0.274. Calibration takes **≈ 4 s** on
CPU (Intel Xeon, no GPU required). Mean over 3 seeds (paper): CDQ 62.6±1.2, CWB 62.9±5.7,
GQH 58.3±1.9.

The official CDQ-calibrated weights are released at `few_shot/ckpt/exp3_best_CDQ.pt`.
`few_shot/train_few_shot.py` is the head-only v1 baseline (no BN adaptation, no LoRA).

### Method ablation (query-set accuracy, %)

| Method | CDQ seed42 | Description |
|---|---|---|
| `head_only` | – | freeze backbone, tune classification head only (BN implicitly drifted) |
| `bn_head` | – | explicit BN-statistics adaptation (10 ep) + head |
| `lora_head` | **63.57** | BN adaptation + head + LoRA(r=8) — final method |

## Data layout (not included)

Prepare two directories with the structure below and point the scripts at them via the
`TAC_*` environment variables (defaults are `<repo>/data/...`):

```
dataset_deep_pre_0817/            # source domain (train): OY LG YZ ZSY
dataset_deep_pre_three_0817/      # target domain (test):  CDQ CWB GQH
  labels.csv        # columns: data_name, person, material
  stats.json        # {"mu": [...], "sigma": [...]} per-channel whitening stats (Fx, Fy, Fz)
  <person>/*.npy    # per-sample pre-whitened force time series [150, 3] float32
```

| Env var | Used by | Default |
|---|---|---|
| `TAC_SRC_DIR` | classifier | `<repo>/data/dataset_deep_pre_0817` |
| `TAC_TGT_DIR` | classifier + few-shot | `<repo>/data/dataset_deep_pre_three_0817` |
| `TAC_OUT_DIR` | classifier | `<repo>/outputs/exp5` |
| `TAC_BACKBONE` | few-shot v2 | `<repo>/checkpoints/backbone_3person_OY_LG_YZ.pt` |

## Requirements

- Python ≥ 3.8, `torch ≥ 2.0`, `numpy`. CPU-only is fine (few-shot calibration ≈ 4 s; the
  classifier trains in a few minutes on CPU).

## Reproducibility notes

- Support-set sampling is stratified and seed-fixed (`np.random.default_rng(seed)`).
- Periodic evaluations iterate the query DataLoader, which consumes global RNG — part of the
  official protocol; keep the scripts unmodified when reproducing paper numbers.

## License

For academic use. Contact the authors for commercial use.
