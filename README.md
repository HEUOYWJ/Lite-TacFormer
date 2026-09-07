# Lite-TacFormer

Tactile material recognition with a lightweight Transformer: a **source-domain classifier** trained on multiple operators plus an **ultra-fast few-shot personal calibration** protocol (BN statistics adaptation + classification head + LoRA adapters) that adapts a pretrained backbone to a new operator using only **20 samples (5 per material)**, trained **on CPU in ~4 seconds**.

Companion code of the ICASSP 2027 submission. The dataset is **not** distributed in this repository.

## Sensor & Data

- **Sensor**: PaXini PX6AX-GEN3-CP-L5325-Omega tactile sensing array — 239 taxels × 3-axis (normal force `fz` and shear forces `fx`/`fy`), USB interface.
## License

For academic use. Contact the authors for commercial use.
