# -*- coding: utf-8 -*-
"""
决定性复验（只读）：纠正 diag_threshold_control.py 的变量错配。

背景
----
上一版对照脚本（diag_threshold_control.py:64-65）这样构造模型：
    m.encoder = HDCPointLevelEncoder(m.dimensions, 128)   # ← 全新随机编码
    m.models.load_state_dict(ck['models'])                # ← 只加载原型
编码器内部的 value/channel/position 超向量是随机初始化的，**与训练时的编码器不是同一组随机向量**。
用「A 编码」训练出的原型去和「B 编码」产出的样本算相似度，得到的相似度没有任何语义。
上一版由此得出的 "通道8 判据通过 1288 / margin 是全灭开关" 需要推翻重验。

本脚本对照两个变量：
  A. encoder 随机初始化（复现上一版的错配）        —— 预期复现 1288 量级
  B. encoder 从 checkpoint 加载（= 真实模型行为）  —— 与原型同源，才是真实读数
  并且对 B 直接把整条链路跑到 Total Pred，回答「0 到底断在判据还是断在写回」。

不写任何模型/数据文件；输出到 _sweep3_out.txt（独立于上一轮的 _sweep2_out.txt）。
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


from hdc_point_level_classifier import HDCPointLevelEncoder
from test_hdc_unified import HDCUnifiedDataset, HDCUnifiedModel, CHANNELS
from hdc_utils import process_prediction_results, calculate_metrics

DATA_DIR = r'D:/HDC/data/test'
MODEL_PATH = r'D:/HDC/models/hdc_point_level.pth'
TARGET = '180123_17113片段.xxc'

ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
log("checkpoint keys :", list(ck.keys()))
log("dimensions      :", ck.get('dimensions'), " channels:", ck.get('channels'))
log("含 encoder ?     :", ck.get('encoder') is not None)
log("")

hdc_file = f'{DATA_DIR}/{TARGET}_hdc.csv'
ds = HDCUnifiedDataset(DATA_DIR, is_train=False, files=[hdc_file], channels=CHANNELS)
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
df_hdc = pd.read_csv(hdc_file)
df_label = pd.read_csv(f'{DATA_DIR}/{TARGET}_label.csv')
label_cols = [f'label_{ch}' for ch in CHANNELS]
merged = pd.merge(df_hdc[['encoding']], df_label[['encoding'] + label_cols],
                  on='encoding', how='left').fillna(0)
y_true = merged[label_cols].values

log("=" * 78)
log(f"目标文件 {TARGET}   窗口样本 {len(ds)}   文件行数 {len(df_hdc)}   真值点 {int((y_true > 0).sum())}")
log("=" * 78)


def build(load_encoder):
    m = HDCUnifiedModel(device='cpu', channels=CHANNELS)
    m.dimensions = ck['dimensions']
    m.channels = ck['channels']
    m.num_channels = len(ck['channels'])
    m.is_single_channel = m.num_channels == 1
    m.encoder = HDCPointLevelEncoder(m.dimensions, 128)
    if load_encoder:
        m.encoder.load_state_dict(ck['encoder'])   # ★ 上一版漏掉的就是这一行
    m.models.load_state_dict(ck['models'])
    return m


def probe(tag, load_encoder):
    m = build(load_encoder)
    with torch.no_grad():
        preds, ns, mds, _ = m.predict(loader, threshold=0.0, margin=0.0)
    preds = np.atleast_2d(preds)

    log(f"--- {tag} ---")
    log(f"  predictions 非零总数      = {int((preds > 0).sum())} / {preds.shape[0]}x{preds.shape[1]}")
    log(f"  逐通道非零预测数          = {[int((preds[:, c] > 0).sum()) for c in range(preds.shape[1])]}")
    log(f"  normal_sim  通道8 [min,max] = [{ns[:, 8].min():.4f}, {ns[:, 8].max():.4f}]  mean={ns[:, 8].mean():.4f}")
    log(f"  max_defect  通道8 [min,max] = [{mds[:, 8].min():.4f}, {mds[:, 8].max():.4f}]  mean={mds[:, 8].mean():.4f}")
    log(f"  通道8 满足 (md>0 且 md>ns)  = {int(((mds[:, 8] > 0) & (mds[:, 8] > ns[:, 8])).sum())}")
    log("")

    # ---- 把链路走完：写入矩阵 → 指标 ----
    y_pred_matrix, _, _, _ = process_prediction_results(
        preds, ns, mds, np.zeros((preds.shape[0], preds.shape[1], 20), dtype=np.float32),
        ds.file_info, df_hdc, CHANNELS)
    y_pred_aligned = y_pred_matrix[:len(y_true)]
    log(f"  写回矩阵 y_pred 非零       = {int((y_pred_aligned > 0).sum())}")
    metrics = calculate_metrics(y_true, y_pred_aligned)
    log(f"  >>> Total True={metrics['total_true']}  Total Pred={metrics['total_pred']} "
        f"TP={metrics['tp']}  P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    log("")
    return metrics


log("=" * 78)
log("[A] encoder 随机初始化（复现上一版错配：编码与原型不同源）")
log("=" * 78)
probe("A 随机编码 + 已训练原型", load_encoder=False)

log("=" * 78)
log("[B] encoder 从 checkpoint 加载（真实模型行为：编码与原型同源）")
log("=" * 78)
probe("B 已训练编码 + 已训练原型", load_encoder=True)

log("=" * 78)
log("读法：")
log("  若 A 的通道8通过数量级 >> B，说明上一版 1288 是『编码-原型不同源』的伪结果。")
log("  若 B 的 Total Pred 仍为 0 且 predictions 非零数=0，则断点在判据本身（相似度太小），")
log("  而非写回链路。")
log("=" * 78)

with open('D:/HDC/_sweep3_out.txt', 'w', encoding='utf-8') as f:
    f.write(LOG.getvalue())
