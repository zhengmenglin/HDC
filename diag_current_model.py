# -*- coding: utf-8 -*-
"""
只读诊断脚本（不训练、不写任何文件、不改模型）
目的：用现有 models/hdc_point_level.pth 对指定测试文件做一次预测，
      量化"Total Pred = 0"的成因——各通道的 normal_sim / max_defect_sim 分布，
      以及判定条件 (max_defect_sim > threshold) & (max_defect_sim > normal_sim + margin) 的通过率。

用法：python diag_current_model.py [不带扩展名的文件名]
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, r'D:/HDC')

from config import config
from hdc_utils import calculate_metrics
from test_hdc_unified import (
    HDCUnifiedDataset, HDCUnifiedModel,
    CHANNELS, IS_SINGLE_CHANNEL, NUM_TYPES,
)

TARGET = sys.argv[1] if len(sys.argv) > 1 else '180123_17113片段.xxc'
DATA_DIR = Path(r'D:/HDC/data/test')
MODEL_PATH = Path(r'D:/HDC/models/hdc_point_level.pth')

THRESHOLD = config.get("test.predict_threshold", 0)
MARGIN = config.get("test.predict_margin", 0)

print("=" * 74)
print(f"诊断文件: {TARGET}")
print(f"通道配置: {CHANNELS}   单通道模式: {IS_SINGLE_CHANNEL}")
print(f"判定参数: threshold={THRESHOLD}  margin={MARGIN}")
print("=" * 74)

hdc_file = DATA_DIR / f"{TARGET}_hdc.csv"
ds = HDCUnifiedDataset(DATA_DIR, is_train=False, files=[hdc_file], channels=CHANNELS)
print(f"滑窗样本数: {len(ds)}")
if len(ds) == 0:
    sys.exit(1)

model = HDCUnifiedModel(device='cpu', channels=CHANNELS)
# 注意：不能用 model.load()——test_hdc_unified.py L486 调用了不存在的 _create_encoder，
# 多通道模式下必然 AttributeError。这里手动完成等价的加载。
from hdc_point_level_classifier import HDCPointLevelEncoder

ck = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
model.dimensions = ck['dimensions']
model.channels = ck['channels']
model.num_channels = len(ck['channels'])
model.is_single_channel = model.num_channels == 1
print(f"[诊断] 模型 dim={model.dimensions} channels={model.channels} 单通道={model.is_single_channel}")
model.encoder = HDCPointLevelEncoder(model.dimensions, 128).to('cpu')
model.encoder.load_state_dict(ck['encoder'])
model.models.load_state_dict(ck['models'])
print("[诊断] encoder + 9 个 Centroid 已手动加载完成")

loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
preds, normal_sims, max_defect_sims, class_sims = model.predict(
    loader, threshold=THRESHOLD, margin=MARGIN
)
preds = np.atleast_2d(preds)

print()
print("=" * 74)
print("预测输出统计")
print("=" * 74)
print(f"predictions.shape = {preds.shape}")
print(f"Total Pred (>0)   = {int((preds > 0).sum())}")
print()

nch = preds.shape[1]
for c in range(nch):
    ch_id = CHANNELS[c] if c < len(CHANNELS) else c
    ns = normal_sims[:, c]
    ds_ = max_defect_sims[:, c]
    p = preds[:, c]
    pass_mask = (ds_ > THRESHOLD) & (ds_ > (ns + MARGIN))
    print(f"通道{ch_id}: 预测非零={int((p>0).sum()):5d}  "
          f"normal_sim[{ns.min():7.4f},{ns.max():7.4f}]  "
          f"max_defect_sim[{ds_.min():7.4f},{ds_.max():7.4f}]  "
          f"判据通过={int(pass_mask.sum()):5d}")

print()
print("=" * 74)
print("真值点位置上的数值（关键：判据为什么没通过）")
print("=" * 74)

df_hdc = pd.read_csv(hdc_file)
enc_to_row = {int(e): i for i, e in enumerate(df_hdc['encoding'].values)}
# 收集每个预测样本的 (matrix_row, center_encoding)
rows = []
for idx, info in enumerate(ds.file_info):
    er = enc_to_row.get(int(info['center_encoding']))
    if er is not None:
        rows.append((er, idx))

# 真值
label_file = DATA_DIR / f"{TARGET}_label.csv"
df_label = pd.read_csv(label_file)
label_cols = [f'label_{ch}' for ch in CHANNELS]
merged = pd.merge(df_hdc[['encoding']], df_label[['encoding'] + label_cols],
                  on='encoding', how='left').fillna(0)
y_true = merged[label_cols].values

y_pred = np.zeros_like(y_true)
for er, idx in rows:
    y_pred[er] = preds[idx]

pos_rows = [r for r in range(len(y_true)) if (y_true[r] > 0).any()]
print(f"真值点行数: {len(pos_rows)}")
for r in pos_rows[:15]:
    line = f"  row={r:5d} enc={int(df_hdc['encoding'].values[r]):6d} 真值={list(y_true[r])}"
    print(line)
    # 找出该行对应的样本索引
    sidx = next((i for (er, i) in rows if er == r), None)
    if sidx is None:
        print("        -> 该行没有对应预测样本（未进入滑窗中心）")
        continue
    ns = normal_sims[sidx]
    dd = max_defect_sims[sidx]
    pr = preds[sidx]
    print(f"        -> normal_sim={np.round(ns,4).tolist()}")
    print(f"           max_defect_sim={np.round(dd,4).tolist()}")
    print(f"           pred={pr.tolist()}")

print()
print("=" * 74)
print("整体指标（照 hdc_utils.calculate_metrics 口径：真值>0 视为正）")
print("=" * 74)
m = calculate_metrics(y_true, y_pred)
for k in ['precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total_true', 'total_pred']:
    print(f"  {k:12s}: {m[k]}")
