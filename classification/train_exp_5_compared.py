# -*- coding: utf-8 -*-
"""
实验 5: 零样本表示对比实验 (6 种输入表示, 4 人训练 -> 3 人测试)
  动机: few-shot 定稿前尝试过 6 种输入表示做跨人零样本分类, 全部失败 (30-33%);
        现按论文对比实验要求统一协议重跑, 每种表示输出
        总体 + 每材质 (iron/paper/plastic/pu) 准确率.
  训练: dataset_deep_pre_0817 的 OY + LG + YZ + ZSY 全量 (4069 条)
  测试: dataset_deep_pre_three_0817 的 CDQ / CWB / GQH (2167 条, 采集协议不同)
  协议: 与 exp_2 (train_exp_2.py) 完全一致:
        50ep / batch 32 / lr 1e-3 / AdamW wd 1e-4 / CosineAnnealing
        无验证集选择 (best = 最终 epoch 权重, 不用测试集选模型)
        每 10ep 打印测试轨迹 (evaluate 迭代 DataLoader 消耗全局 RNG, 是协议一部分)
  表示 (在 Dataset.__getitem__ 中施加):
    peaknorm_norm    样本级峰值力归一化 x/(max|x|+1e-8)          [150,3]
    peaknorm_nonorm  原始白化域时域 (对照, 应复现 exp_2)          [150,3]
    fft              rfft 振幅谱 (相位丢弃)                       [76,3]
    diff             一阶差分 (砍 DC)                             [149,3]
    friction         白化域直接算摩擦代理 (F_shear, Fz, mu)        [150,3]
    friction_phys    反白化->纯物理摩擦代理->训练集统计再白化      [150,3]
  输出 (本目录, 自包含):
    {repr}/best.pt + results.json   (3 人 + 合并池, 总体+每类+混淆矩阵)
    summary.json                    汇总表
  验证: peaknorm_nonorm 应复现 exp_2: CDQ 36.94 / CWB 25.17 / GQH 40.14 / 合并 34.06

数据布局 (repo 外, 用环境变量指定):
    TAC_SRC_DIR  源域训练数据目录, 默认 <repo>/data/dataset_deep_pre_0817
    TAC_TGT_DIR  目标域测试数据目录, 默认 <repo>/data/dataset_deep_pre_three_0817
    TAC_OUT_DIR  输出目录, 默认 <repo>/outputs/exp5
  数据集 (每目录含 labels.csv + stats.json + <person>/*.npy) 不随仓库分发,
  需按 ICASSP2027 采集协议自行准备 (见 README)。

用法 (CPU):
    python train_exp_5_compared.py --repr peaknorm_nonorm
    python train_exp_5_compared.py --repr all
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
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]     # 仓库根目录 (含 exp_class.py)
sys.path.insert(0, str(REPO))
import exp_class  # BaselineTactileTransformer

MATERIALS = ["iron", "paper", "plastic", "pu"]
TRAIN_PERSONS = ["OY", "LG", "YZ", "ZSY"]
TEST_PERSONS = ["CDQ", "CWB", "GQH"]
SEED = 42
SRC_DIR = os.environ.get("TAC_SRC_DIR",
                         str(REPO / "data" / "dataset_deep_pre_0817"))
TGT_DIR = os.environ.get("TAC_TGT_DIR",
                         str(REPO / "data" / "dataset_deep_pre_three_0817"))
OUT_DIR = Path(os.environ.get("TAC_OUT_DIR", str(REPO / "outputs" / "exp5")))

REPRS = ["peaknorm_norm", "peaknorm_nonorm", "fft", "diff",
         "friction", "friction_phys"]
MAX_LEN = {"peaknorm_norm": 150, "peaknorm_nonorm": 150, "fft": 76,
           "diff": 149, "friction": 150, "friction_phys": 150}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_stats(data_dir):
    """全局白化统计 (通道顺序 Fx, Fy, Fz)."""
    with open(Path(data_dir) / "stats.json", encoding="utf-8") as f:
        s = json.load(f)
    return (torch.tensor(s["mu"], dtype=torch.float32),
            torch.tensor(s["sigma"], dtype=torch.float32))


def compute_proxy_stats(X, mu, sigma):
    """训练集上: 反白化 -> 摩擦代理三通道 -> 统计 (friction_phys 再白化用)."""
    x = X * sigma + mu                                    # [N,150,3] 牛顿
    f_shear = torch.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
    fric = f_shear / (x[:, :, 2].abs() + 1e-8)
    xp = torch.stack([f_shear, x[:, :, 2], fric], dim=-1)  # [N,150,3] 代理
    return xp.mean(dim=(0, 1)), xp.std(dim=(0, 1)) + 1e-8


def make_dataset(data_dir, persons, repr_name, proxy_stats=None):
    """按人过滤 deep_pre 数据, 返回 (X, y) 张量 (表示在 __getitem__ 施加)."""

    class ReprDataset(Dataset):
        def __init__(self):
            rows = []
            with open(Path(data_dir) / "labels.csv", encoding="utf-8") as f:
                for r in csv.reader(f):
                    if r and r[0] != "data_name" and r[1] in persons:
                        rows.append(r)
            self.names = [r[0] for r in rows]
            self.y = torch.tensor([MATERIALS.index(r[2]) for r in rows],
                                  dtype=torch.long)
            x_list = [np.load(Path(data_dir) / r[1] / (r[0] + ".npy"))
                      for r in rows]
            self.X = torch.from_numpy(np.stack(x_list)).float()  # [N,150,3]
            self.repr = repr_name
            if repr_name == "friction_phys":
                self.mu, self.sigma = load_stats(data_dir)
                self.pmu, self.psigma = proxy_stats

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x = self.X[i]                                  # [150,3]
            if self.repr == "peaknorm_norm":
                # 样本级峰值力归一化: 抹掉每条样本的绝对力度
                x = x / (torch.max(torch.abs(x)) + 1e-8)
            elif self.repr == "fft":
                # 频域振幅谱 (沿时间 dim=0, 150 -> 76 频率点, 相位丢弃)
                x = torch.abs(torch.fft.rfft(x, dim=0))
            elif self.repr == "diff":
                # 一阶差分: 砍绝对力度 (DC), 150 -> 149
                x = torch.diff(x, dim=0)
            elif self.repr in ("friction", "friction_phys"):
                # 动态摩擦系数代理: F_shear / |F_z|
                if self.repr == "friction_phys":
                    x = x * self.sigma + self.mu           # 反白化 -> 牛顿
                F_x, F_y, F_z = x[:, 0], x[:, 1], x[:, 2]
                f_shear = torch.sqrt(F_x ** 2 + F_y ** 2)
                mu = f_shear / (torch.abs(F_z) + 1e-8)
                x = torch.stack([f_shear, F_z, mu], dim=1)  # [150,3]
                if self.repr == "friction_phys":
                    x = (x - self.pmu) / self.psigma       # 代理通道再白化
            return x, self.y[i]

    return ReprDataset()


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_class = torch.zeros(len(MATERIALS), dtype=torch.long)
    per_class_total = torch.zeros(len(MATERIALS), dtype=torch.long)
    cm = np.zeros((len(MATERIALS), len(MATERIALS)), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
                per_class_total[t] += 1
                if t == p:
                    per_class[t] += 1
                cm[t][p] += 1
    acc = correct / total if total else 0.0
    return acc, cm, per_class.tolist(), per_class_total.tolist()


def run_repr(repr_name, device):
    """与 exp_2 逐行一致的训练 + 评估 (仅表示与 max_seq_len 不同)."""
    set_seed()
    out_dir = OUT_DIR / repr_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- friction_phys 的代理通道统计 (训练集) ----
    proxy_stats = None
    if repr_name == "friction_phys":
        mu, sigma = load_stats(SRC_DIR)
        rows = []
        with open(Path(SRC_DIR) / "labels.csv", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and r[0] != "data_name" and r[1] in TRAIN_PERSONS:
                    rows.append(r)
        x_list = [np.load(Path(SRC_DIR) / r[1] / (r[0] + ".npy"))
                  for r in rows]
        X_tr = torch.from_numpy(np.stack(x_list)).float()
        pmu, psigma = compute_proxy_stats(X_tr, mu, sigma)
        proxy_stats = (pmu, psigma)
        print("代理通道统计(训练集): mean %s std %s"
              % (pmu.round(decimals=3).tolist(),
                 psigma.round(decimals=3).tolist()))

    # ---- 数据 ----
    ds_train = make_dataset(SRC_DIR, TRAIN_PERSONS, repr_name, proxy_stats)
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=False)
    tests = {}
    for p in TEST_PERSONS:
        tests[p] = DataLoader(make_dataset(TGT_DIR, [p], repr_name, proxy_stats),
                              batch_size=32, num_workers=2, pin_memory=True)
    print("[%s] train %d (%s) | test %s (%d 条)" %
          (repr_name, len(ds_train), "+".join(TRAIN_PERSONS),
           "+".join(TEST_PERSONS),
           sum(len(dl.dataset) for dl in tests.values())))

    # ---- 模型: 表示决定 max_seq_len ----
    model = exp_class.BaselineTactileTransformer(
        input_channels=3, num_classes=len(MATERIALS),
        max_seq_len=MAX_LEN[repr_name]).to(device)
    print("  model: max_seq_len=%d, params=%d"
          % (MAX_LEN[repr_name], sum(p.numel() for p in model.parameters())))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # ---- 训练 (每 10ep 打印轨迹, 不做模型选择) ----
    history = []
    t0 = time.time()
    for epoch in range(1, 51):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()
        train_acc = correct / total
        row = {"epoch": epoch, "loss": running_loss / total,
               "train_acc": train_acc}
        if epoch % 10 == 0 or epoch == 1:
            for p, dl in tests.items():
                acc, *_ = evaluate(model, dl, device)
                row["test_%s" % p] = acc
            print("  ep %3d | loss %.4f | train %.4f | "
                  % (epoch, running_loss / total, train_acc)
                  + " ".join("%s %.3f" % (p, row["test_%s" % p])
                             for p in TEST_PERSONS))
        history.append(row)
    print("  train done in %.0fs" % (time.time() - t0))

    # ---- 最终评估 (最终 epoch 权重) ----
    torch.save(model.state_dict(), out_dir / "best.pt")
    results = {"repr": repr_name, "history": history,
               "train_persons": TRAIN_PERSONS, "test_persons": TEST_PERSONS,
               "n_train": len(ds_train)}
    for p, dl in tests.items():
        acc, cm, pc, pct = evaluate(model, dl, device)
        results[p] = {"acc": acc, "confusion": cm.tolist(),
                      "per_class_correct": pc, "per_class_total": pct}
        print("  %s: acc %.4f | %s"
              % (p, acc, " ".join("%s %.3f" % (MATERIALS[i], pc[i] / pct[i])
                                  for i in range(4))))
    all_ds = torch.utils.data.ConcatDataset(
        [tests[p].dataset for p in TEST_PERSONS])
    loader = torch.utils.data.DataLoader(all_ds, batch_size=64,
                                         num_workers=2, pin_memory=True)
    acc, cm, pc, pct = evaluate(model, loader, device)
    results["all"] = {"acc": acc, "confusion": cm.tolist(),
                      "per_class_correct": pc, "per_class_total": pct}
    print("  merged: acc %.4f | %s"
          % (acc, " ".join("%s %.3f" % (MATERIALS[i], pc[i] / pct[i])
                           for i in range(4))))

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("  saved -> %s/best.pt + results.json" % out_dir)
    return results


def main():
    ap = argparse.ArgumentParser(description="实验5: 零样本表示对比")
    ap.add_argument("--repr", default="all",
                    help="表示名或 all (%s)" % "/".join(REPRS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("device: %s (%s)" % (device,
                               "CPU 模式" if device.type == "cpu"
                               else torch.cuda.get_device_name(0)))
    reprs = REPRS if args.repr == "all" else [args.repr]
    for rn in reprs:
        assert rn in REPRS, "未知表示: %s" % rn

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_res = {}
    for rn in reprs:
        all_res[rn] = run_repr(rn, device)

    # ---- 汇总表 ----
    print("\n===== 汇总: 总体准确率 (%) =====")
    print("%-16s | %-8s | %-8s | %-8s | %-8s" %
          ("表示", "CDQ", "CWB", "GQH", "合并"))
    summary = {}
    for rn in reprs:
        r = all_res[rn]
        summary[rn] = {p: round(r[p]["acc"] * 100, 2) for p in TEST_PERSONS}
        summary[rn]["all"] = round(r["all"]["acc"] * 100, 2)
        print("%-16s | %-8.1f | %-8.1f | %-8.1f | %-8.1f"
              % (rn, summary[rn]["CDQ"], summary[rn]["CWB"],
                 summary[rn]["GQH"], summary[rn]["all"]))
    print("\n===== 合并池每类准确率 (%) =====")
    print("%-16s | %-8s | %-8s | %-8s | %-8s" %
          ("表示", "iron", "paper", "plastic", "pu"))
    for rn in reprs:
        pc, pt = (all_res[rn]["all"]["per_class_correct"],
                  all_res[rn]["all"]["per_class_total"])
        per = [pc[i] / pt[i] * 100 for i in range(4)]
        summary[rn]["per_class_all"] = {MATERIALS[i]: round(per[i], 2)
                                        for i in range(4)}
        print("%-16s | %-8.1f | %-8.1f | %-8.1f | %-8.1f"
              % (rn, per[0], per[1], per[2], per[3]))
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "与 exp_2 一致 (50ep/lr1e-3/Cosine/最终epoch权重)",
                   "summary": summary}, f, ensure_ascii=False, indent=1)
    print("\nsaved -> %s/summary.json" % OUT_DIR)


if __name__ == "__main__":
    main()
