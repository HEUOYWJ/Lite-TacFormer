# -*- coding: utf-8 -*-
"""
Few-Shot 个性化校准 v2 (升级版): BN 统计量校准 + 分类头 + LoRA 小适配器
  校准协议: 每材质 5 条 (20 条) 支持集, 查询集 = 该人其余数据
  方法三档:
    head_only : 冻结主干, 只训分类头 (v1 基准, BN 隐式漂移)
    bn_head   : 显式 BN 统计量适应 (无梯度前向 10 epoch) + 训分类头
    lora_head : BN 适应 + 分类头 + LoRA(r=8) 适配器 (out_proj + FFN)
  测试: CDQ / CWB / GQH 三人 (支持集各自从本人数据抽取, 其余为查询集)

用法 (CPU):
    python train_few_shot_upgrade.py --method lora_head --test-person CDQ --seed 42

  数据/权重:
    TAC_TGT_DIR   测试人数据目录 (默认 <repo>/data/dataset_deep_pre_three_0817)
    TAC_BACKBONE  源域主干权重 (默认 <repo>/checkpoints/backbone_3person_OY_LG_YZ.pt)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_few_shot import (TactileDataset, build_support_query,  # noqa: E402
                            MATERIALS)

SEED = 42
REPO = Path(__file__).resolve().parents[1]     # 仓库根目录 (含 exp_class.py)
BACKBONE_RAW = os.environ.get(
    "TAC_BACKBONE", str(REPO / "checkpoints" / "backbone_3person_OY_LG_YZ.pt"))
TGT_DATA = os.environ.get(
    "TAC_TGT_DIR", str(REPO / "data" / "dataset_deep_pre_three_0817"))


class LoRALinear(nn.Module):
    """LoRA 包装: 冻结原 Linear, 附加低秩 B@A 增量. y = Wx + (x@A^T@B^T)*scaling."""

    def __init__(self, linear, r=8, alpha=8):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        return self.linear(x) + torch.matmul(
            torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * self.scaling


def apply_lora(model, r=8):
    """在 transformer 每层 FFN 的两个 Linear 上挂 LoRA.
    两点注意:
    1. 不包装 attention 的 out_proj —— torch 2.2 的 MHA 在 need_weights=False
       时直接属性访问 out_proj.weight 走融合路径, 包装会崩且增量被绕过;
    2. FFN 融合快路径会直接读 linear1/linear2.weight (eval 且无梯度时触发),
       必须把 activation_relu_or_gelu 置 0 让 torch 退回标准模块调用路径
       (数学完全相同), 否则 eval 时 LoRA 增量被静默丢弃."""
    n = 0
    for layer in model.transformer_encoder.layers:
        layer.linear1 = LoRALinear(layer.linear1, r)
        layer.linear2 = LoRALinear(layer.linear2, r)
        layer.activation_relu_or_gelu = 0   # 禁用融合快路径, 保住 LoRA 生效
        n += 2
    return n


def adapt_bn(model, dl_sup, epochs=10, device="cpu"):
    """BN 统计量校准: train 模式无梯度前向, 让 running stats 贴向新人域."""
    model.train()
    for _ in range(epochs):
        for inputs, _ in dl_sup:
            with torch.no_grad():
                model(inputs.to(device))


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_c = torch.zeros(4, dtype=torch.long)
    per_t = torch.zeros(4, dtype=torch.long)
    cm = torch.zeros((4, 4), dtype=torch.long)
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
    ap = argparse.ArgumentParser(description="Few-Shot 校准 v2 (BN+头+LoRA)")
    ap.add_argument("--method", choices=["head_only", "bn_head", "lora_head"],
                    default="lora_head")
    ap.add_argument("--test-person", choices=["CDQ", "CWB", "GQH"], default="CDQ")
    ap.add_argument("--support-per-class", type=int, default=5)
    ap.add_argument("--calib-epochs", type=int, default=50)
    ap.add_argument("--calib-lr", type=float, default=0.005)
    ap.add_argument("--bn-adapt-epochs", type=int, default=10)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    print("method=%s | test=%s | seed=%d | 支持集 %d 条/人"
          % (args.method, args.test_person, args.seed, args.support_per_class * 4))

    # ---- 数据: 支持集 + 查询集 ----
    ds_all = TactileDataset(
        TGT_DATA, [args.test_person], "raw")
    sup_idx, qry_idx = build_support_query(ds_all, args.support_per_class, args.seed)
    dl_sup = DataLoader(Subset(ds_all, sup_idx), batch_size=20, shuffle=True)
    dl_qry = DataLoader(Subset(ds_all, qry_idx), batch_size=64, shuffle=False)
    print("%s: 支持集 %d | 查询集 %d" % (args.test_person, len(sup_idx), len(qry_idx)))

    # ---- 加载源域主干 ----
    sys.path.insert(0, str(REPO))
    from exp_class import BaselineTactileTransformer
    model = BaselineTactileTransformer(input_channels=3, num_classes=4)
    model.load_state_dict(torch.load(BACKBONE_RAW, map_location=device))
    model.to(device)

    if args.method == "lora_head":
        n_lora = apply_lora(model, args.lora_rank)
        model.to(device)   # LoRA 新参数移到目标设备
        print("LoRA 已挂载: %d 个模块 (r=%d)" % (n_lora, args.lora_rank))

    # ---- 冻结主干, 仅保留待校准参数 ----
    for param in model.parameters():
        param.requires_grad = False
    trainable = []
    for param in model.classifier.parameters():
        param.requires_grad = True
        trainable.append(param)
    if args.method == "lora_head":
        for module in model.modules():
            if isinstance(module, LoRALinear):
                for param in module.lora_A, module.lora_B:
                    param.requires_grad = True
                    trainable.append(param)
    n_train = sum(p.numel() for p in trainable)
    print("可训练参数: %d (主干全部冻结)" % n_train)

    # ---- 校准前零样本 ----
    pre_acc, *_ = evaluate(model, dl_qry, device)

    # ---- BN 统计量适应 (bn_head / lora_head 显式阶段) ----
    if args.method in ("bn_head", "lora_head"):
        adapt_bn(model, dl_sup, args.bn_adapt_epochs, device)
        print("BN 统计量适应完成 (%d epoch, 无梯度)" % args.bn_adapt_epochs)

    # ---- 分类头 (+LoRA) 校准 ----
    optimizer = torch.optim.AdamW(trainable, lr=args.calib_lr)
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
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
        if epoch % 10 == 0 or epoch == 1:
            q_acc, *_ = evaluate(model, dl_qry, device)
            print("  calib ep %3d | loss %.4f | query %.4f"
                  % (epoch, loss_sum / len(dl_sup), q_acc))
    print("校准完成 (%.1fs)" % (time.time() - t0))

    # ---- 查询集评估 ----
    acc, cm, pc, pt = evaluate(model, dl_qry, device)
    print("\n=== %s 查询集 (%d 条) method=%s seed=%d ==="
          % (args.test_person, len(qry_idx), args.method, args.seed))
    print("统一: 校准前 %.4f -> 校准后 %.4f (Δ %+.4f)"
          % (pre_acc, acc, acc - pre_acc))
    print("每材质: " + ", ".join("%s %.3f (%d/%d)" % (MATERIALS[i], pc[i] / pt[i],
          pc[i], pt[i]) for i in range(4)))
    print("混淆矩阵 (行=真实 iron/paper/plastic/pu):")
    print("   " + " ".join("%6s" % m for m in MATERIALS))
    for i, row in enumerate(cm.tolist()):
        print("  %-7s %s" % (MATERIALS[i], " ".join("%6d" % v for v in row)))

    out_dir = REPO / "outputs" / "few_shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / ("results_%s_%s_seed%d.json"
                         % (args.method, args.test_person, args.seed)),
              "w", encoding="utf-8") as f:
        json.dump({"method": args.method, "test_person": args.test_person,
                   "seed": args.seed, "pre_acc": pre_acc, "test_acc": acc,
                   "per_class_correct": pc, "per_class_total": pt,
                   "confusion": cm.tolist()}, f, ensure_ascii=False, indent=1)
    print("\nsaved -> %s/results_%s_%s_seed%d.json"
          % (out_dir, args.method, args.test_person, args.seed))


if __name__ == "__main__":
    main()
