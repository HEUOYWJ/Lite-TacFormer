# -*- coding: utf-8 -*-
"""
Few-Shot 个性化校准极速验证 (插拔式校准故事)
  每次换操作员/夹爪, 只需 5 秒采集 20 条数据 (每材质 5 条) 即可适配
  1. 支持集: CDQ 每材质随机抽 5 条 (共 20), 其余为查询集
  2. 冻结源域预训练主干 (OY+LG+YZ), 仅解冻分类头
  3. 极速微调: 20 条数据上 50 epoch (lr=0.005)
  4. 查询集盲测: 校准前后对比 (同一查询集), 输出统一/每材质/混淆矩阵

用法 (CPU):
    python train_few_shot.py --backbone raw --seed 42
    python train_few_shot.py --backbone friction

  数据目录环境变量: TAC_TGT_DIR
      (默认 <repo>/data/dataset_deep_pre_three_0817, 数据集不随仓库分发)
  主干权重默认在 <repo>/checkpoints/: 仓库仅附带 raw 主干
      (backbone_3person_OY_LG_YZ.pt); friction/peaknorm 需先用
      classification/train_exp_5_compared.py 自行训练后放入同目录
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

REPO = Path(__file__).resolve().parents[1]     # 仓库根目录 (含 exp_class.py)
sys.path.insert(0, str(REPO))
from exp_class import BaselineTactileTransformer  # noqa: E402

MATERIALS = ["iron", "paper", "plastic", "pu"]
SEED = 42
TGT_DATA = os.environ.get(
    "TAC_TGT_DIR", str(REPO / "data" / "dataset_deep_pre_three_0817"))
TEST_PERSON = "CDQ"
CKPT_DIR = REPO / "checkpoints"
# 预训练主干 (均训练于 OY+LG+YZ, 架构一致 max_seq_len=150)
BACKBONES = {
    "raw": (CKPT_DIR / "backbone_3person_OY_LG_YZ.pt", "raw"),
    "friction": (CKPT_DIR / "exp_friction_best.pt", "friction"),
    "peaknorm": (CKPT_DIR / "exp_peaknorm_norm_best.pt", "peaknorm"),
}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TactileDataset(Dataset):
    """CDQ 全量 + 输入变换 (与预训练主干训练时一致)."""

    def __init__(self, data_dir, persons, transform="raw"):
        rows = []
        with open(Path(data_dir) / "labels.csv", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and r[0] != "data_name" and r[1] in persons:
                    rows.append(r)
        self.y = torch.tensor([MATERIALS.index(r[2]) for r in rows], dtype=torch.long)
        x_list = [np.load(Path(data_dir) / r[1] / (r[0] + ".npy")) for r in rows]
        self.X = torch.from_numpy(np.stack(x_list)).float()  # [N, 150, 3] 白化域
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.transform == "peaknorm":
            peak_val = torch.max(torch.abs(x))
            x = x / (peak_val + 1e-8)
        elif self.transform == "friction":
            F_x, F_y, F_z = x[:, 0], x[:, 1], x[:, 2]
            F_shear = torch.sqrt(F_x ** 2 + F_y ** 2)
            friction_coeff = F_shear / (torch.abs(F_z) + 1e-8)
            x = torch.stack([F_shear, F_z, friction_coeff], dim=1)
        return x, self.y[i]


def build_support_query(ds, k_per_class=5, seed=SEED):
    """每材质随机抽 k 条做支持集, 其余为查询集 (分层, 可复现)."""
    rng = np.random.default_rng(seed)
    support_idx, query_idx = [], []
    for c in range(4):
        idx = np.where(ds.y.numpy() == c)[0]
        rng.shuffle(idx)
        support_idx.extend(idx[:k_per_class].tolist())
        query_idx.extend(idx[k_per_class:].tolist())
    return support_idx, query_idx


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_c = torch.zeros(4, dtype=torch.long)
    per_t = torch.zeros(4, dtype=torch.long)
    cm = np.zeros((4, 4), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.tolist(), pred.tolist()):
                per_t[t] += 1
                if t == p:
                    per_c[t] += 1
                cm[t][p] += 1
    return correct / total, cm, per_c.tolist(), per_t.tolist()


def main():
    ap = argparse.ArgumentParser(description="Few-Shot 个性化校准 (CDQ 支持集->查询集)")
    ap.add_argument("--backbone", choices=sorted(BACKBONES.keys()), default="raw")
    ap.add_argument("--support-per-class", type=int, default=5)
    ap.add_argument("--calib-epochs", type=int, default=50)
    ap.add_argument("--calib-lr", type=float, default=0.005)
    ap.add_argument("--calib-batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    ckpt_path, transform = BACKBONES[args.backbone]
    print("device: %s (%s) | backbone=%s | seed=%d"
          % (device, "CPU 模式" if device.type == "cpu"
             else torch.cuda.get_device_name(0), args.backbone, args.seed))

    # ---- 数据: 支持集 (20) + 查询集 (CDQ 剩余) ----
    ds_all = TactileDataset(TGT_DATA, [TEST_PERSON], transform)
    sup_idx, qry_idx = build_support_query(ds_all, args.support_per_class, args.seed)
    ds_sup = Subset(ds_all, sup_idx)
    ds_qry = Subset(ds_all, qry_idx)
    dl_sup = DataLoader(ds_sup, batch_size=args.calib_batch, shuffle=True)
    dl_qry = DataLoader(ds_qry, batch_size=64, shuffle=False)
    print("CDQ 全量 %d | 支持集 %d (每材质 %d 条) | 查询集 %d"
          % (len(ds_all), len(ds_sup), args.support_per_class, len(ds_qry)))
    print("支持集标签分布: " + ", ".join(
        "%s %d" % (MATERIALS[c], sum(ds_all.y[i] == c for i in sup_idx))
        for c in range(4)))

    # ---- 加载源域预训练模型, 冻结主干, 仅解冻分类头 ----
    model = BaselineTactileTransformer(input_channels=3, num_classes=4)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("已加载 %s 并冻结主干 (仅分类头 %d 参数可训练)"
          % (ckpt_path.split("/")[-2], sum(p.numel() for p in
                                           model.classifier.parameters())))

    # ---- 校准前零样本评估 (同一查询集) ----
    pre_acc, pre_cm, pre_pc, pre_pt = evaluate(model, dl_qry, device)
    print("\n[校准前] 查询集准确率: %.4f (%d/%d) | %s"
          % (pre_acc, int(round(pre_acc * len(ds_qry))), len(ds_qry),
             " ".join("%s %.3f" % (MATERIALS[i], pre_pc[i] / pre_pt[i])
                      for i in range(4))))

    out_dir = REPO / "outputs" / "few_shot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 极速微调 (仅分类头) ----
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.calib_lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    t0 = time.time()
    best_acc, best_epoch = 0.0, -1
    for epoch in range(1, args.calib_epochs + 1):
        model.train()
        loss_sum = 0.0
        for inputs, labels in dl_sup:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        q_acc, *_ = evaluate(model, dl_qry, device) \
            if epoch % 10 == 0 or epoch == 1 else (None, None, None, None)
        history.append({"epoch": epoch, "loss": loss_sum / len(dl_sup),
                        "query_acc": q_acc})
        if q_acc is not None and q_acc > best_acc:
            best_acc, best_epoch = q_acc, epoch
            torch.save(model.state_dict(), out_dir / "best_calib.pt")
        if epoch % 10 == 0 or epoch == 1:
            print("  calib epoch %3d | loss %.4f | query %.4f"
                  % (epoch, loss_sum / len(dl_sup),
                     q_acc if q_acc is not None else -1.0))
    print("校准完成 (%.1fs)" % (time.time() - t0))

    # ---- 校准后评估 (最终模型) ----
    acc, cm, pc, pt = evaluate(model, dl_qry, device)
    print("\n=== CDQ 查询集 (%d 条, 校准后) backbone=%s seed=%d ==="
          % (len(ds_qry), args.backbone, args.seed))
    print("统一准确率: %.4f (%d/%d) | 校准前 %.4f -> 校准后 %.4f (Δ %+.4f)"
          % (acc, int(round(acc * len(ds_qry))), len(ds_qry),
             pre_acc, acc, acc - pre_acc))
    print("每材质: " + ", ".join("%s %.3f (%d/%d)" % (MATERIALS[i], pc[i] / pt[i],
          pc[i], pt[i]) for i in range(4)))
    print("混淆矩阵 (行=真实 iron/paper/plastic/pu):")
    print("   " + " ".join("%6s" % m for m in MATERIALS))
    for i, row in enumerate(cm):
        print("  %-7s %s" % (MATERIALS[i], " ".join("%6d" % v for v in row)))

    with open(out_dir / ("results_%s_seed%d.json" % (args.backbone, args.seed)),
              "w", encoding="utf-8") as f:
        json.dump({"backbone": args.backbone, "seed": args.seed,
                   "support_idx": sup_idx,
                   "pre_acc": pre_acc, "test_acc": acc, "best_epoch": best_epoch,
                   "best_acc": best_acc,
                   "per_class_correct": pc, "per_class_total": pt,
                   "confusion": cm.tolist(), "history": history}, f,
                  ensure_ascii=False, indent=1)
    print("\nsaved -> %s/results_%s_seed%d.json + best_calib.pt"
          % (out_dir, args.backbone, args.seed))


if __name__ == "__main__":
    main()
