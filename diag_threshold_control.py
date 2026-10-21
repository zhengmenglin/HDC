# -*- coding: utf-8 -*-
"""
决定性对照实验（只读）：
上一轮 randomness 扫描出现反直觉结果——r 增大反而判据通过数下降。
本脚本分离两个变量，定位"通道 0-7 判据通过"的真正来源：
  (1) 是编码（value_hvs）在起作用？
  (2) 还是原型本身就是零向量，判据退化成"负值也算通过"？

对照组：
  A. 原始模型，threshold=0,   margin=0   （= 现状）
  B. 原始模型，threshold=0,   margin=0.02（= 评审报告建议的参数）
  C. 原始模型，threshold=0.1, margin=0.02（抬高门槛）
  D. 打乱原型（把非零原型置乱到别的类）——看预测是否随之改变
  E. 把所有原型强制置零——看是否仍"通过"

不写任何模型/数据文件。
"""
import sys, io
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r'D:/HDC')
LOG = io.StringIO()
def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.write(s + '\n')
    print(s)

from config import config
from hdc_point_level_classifier import HDCPointLevelEncoder
from test_hdc_unified import HDCUnifiedDataset, HDCUnifiedModel, CHANNELS

TARGET = sys.argv[1] if len(sys.argv) > 1 else '180123_17113片段.xxc'
DATA_DIR = r'D:/HDC/data/test'
MODEL_PATH = r'D:/HDC/models/hdc_point_level.pth'

hdc_file = f'{DATA_DIR}/{TARGET}_hdc.csv'
ds = HDCUnifiedDataset(DATA_DIR, is_train=False, files=[hdc_file], channels=CHANNELS)
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

df_hdc = pd.read_csv(hdc_file)
df_label = pd.read_csv(f'{DATA_DIR}/{TARGET}_label.csv')
label_cols = [f'label_{ch}' for ch in CHANNELS]
merged = pd.merge(df_hdc[['encoding']], df_label[['encoding'] + label_cols],
                  on='encoding', how='left').fillna(0)
y_true = merged[label_cols].values
enc_to_row = {int(e): i for i, e in enumerate(df_hdc['encoding'].values)}
rows = [(enc_to_row[int(i['center_encoding'])], idx)
        for idx, i in enumerate(ds.file_info)
        if int(i['center_encoding']) in enc_to_row]
pos_rows = [r for r in range(len(y_true)) if (y_true[r] > 0).any()]
idx_map = {er: idx for er, idx in rows}

ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)


def build(mutate=None):
    m = HDCUnifiedModel(device='cpu', channels=CHANNELS)
    m.dimensions = ck['dimensions']
    m.channels = ck['channels']
    m.num_channels = len(ck['channels'])
    m.is_single_channel = m.num_channels == 1
    m.encoder = HDCPointLevelEncoder(m.dimensions, 128)
    m.models.load_state_dict(ck['models'])
    if mutate:
        mutate(m)
    return m


def run(tag, mutate=None, threshold=0.0, margin=0.0):
    m = build(mutate)
    with torch.no_grad():
        preds, ns, mds, _ = m.predict(loader, threshold=threshold, margin=margin)
    preds = np.atleast_2d(preds)
    y_pred = np.zeros_like(y_true)
    for er, idx in rows:
        y_pred[er] = preds[idx]

    pass_all = int(((mds > threshold) & (mds > ns + margin)).sum())
    # 只看真值点
    sel = [idx_map[r] for r in pos_rows if r in idx_map]
    pos_pass = int(((mds[sel] > threshold) & (mds[sel] > ns[sel] + margin)).sum()) if sel else 0

    # 逐通道判据通过数
    per_ch = []
    for c in range(preds.shape[1]):
        mask = (mds[:, c] > threshold) & (mds[:, c] > ns[:, c] + margin)
        per_ch.append(int(mask.sum()))

    log(f"{tag}")
    log(f"    全样本判据通过 = {pass_all} / {len(preds)}   （其中真值点 {pos_pass} / {len(sel)}）")
    log(f"    真值点 max_defect 均值 = {float(mds[sel].max(axis=1).mean()):.5f}   "
        f"normal 均值 = {float(ns[sel].max(axis=1).mean()):.5f}")
    log(f"    逐通道通过数 = {per_ch}")
    log(f"    全样本 max_defect 全局最大 = {float(mds.max()):.5f}")
    return dict(tag=tag, pass_all=pass_all, pos_pass=pos_pass, per_ch=per_ch)


log("=" * 78)
log(f"决定性对照实验  目标文件={TARGET}   样本数={len(ds)}   真值点={len(pos_rows)}")
log("=" * 78)
log("")

log("--- 对照 A：原始模型 threshold=0 margin=0（现状）---")
run("A 原始 (thr=0,  mar=0)", None, 0.0, 0.0)
log("")
log("--- 对照 B：原始模型 threshold=0 margin=0.02（评审建议）---")
run("B 原始 (thr=0,  mar=0.02)", None, 0.0, 0.02)
log("")
log("--- 对照 C：原始模型 threshold=0.1 margin=0.02（抬高门槛）---")
run("C 原始 (thr=0.1,mar=0.02)", None, 0.1, 0.02)
log("")

log("--- 对照 D：把所有原型强制置零 ---")
def zero_all(m):
    with torch.no_grad():
        for c in range(m.num_channels):
            m.models[c].weight.zero_()
run("D 全零原型 (thr=0, mar=0)", zero_all, 0.0, 0.0)
log("")

log("--- 对照 E：把所有原型置成一个固定的非零向量（非零但不含信息）---")
def const_all(m):
    with torch.no_grad():
        for c in range(m.num_channels):
            m.models[c].weight.fill_(0.01)
run("E 常量原型 (thr=0, mar=0)", const_all, 0.0, 0.0)
log("")

log("=" * 78)
log("结论读法：")
log("  若 D（全零原型）的判据通过数 ≈ A，说明通道 0-7 的'通过'与原型内容无关，")
log("  纯粹是判据在比较两个零向量时的边界行为。")
log("  若 B/C 下通过数骤降到接近 0，说明现状的'通过'是 margin=0 造成的假象。")
log("=" * 78)

with open('D:/HDC/_sweep2_out.txt', 'w', encoding='utf-8') as f:
    f.write(LOG.getvalue())
