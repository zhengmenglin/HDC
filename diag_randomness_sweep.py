# -*- coding: utf-8 -*-
"""
只读实验：扫描 Level 编码的 randomness 参数，看它如何影响
  (A) 相邻级别相似度（编码分辨力）
  (B) 现有模型在真值点上的 max_defect_sim（这决定判据能否翻正）

不训练、不写任何模型/数据文件；只输出到 stdout 和一个文本报告。
用法：
  python diag_randomness_sweep.py [测试文件名] [--re_train]
    默认（不训练）：只用现有 models/hdc_point_level.pth，对每档 randomness
                    重新生成编码器，实测"同一批真值点在同一个已训练原型下的可分性"。
    加 --re_train：每档 randomness 下用 data/train 现场训练一遍，看真实指标。
"""
import sys, io, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r'D:/HDC')

# ---------- 统一写日志（Windows 控制台编码问题规避）----------
LOG = io.StringIO()
def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.write(s + '\n')
    print(s)

from config import config
from hdc_utils import calculate_metrics
from hdc_point_level_classifier import HDCPointLevelEncoder, NUM_LEVELS
from test_hdc_unified import (
    HDCUnifiedDataset, HDCUnifiedModel, CHANNELS, NUM_TYPES,
)

TARGET = '180123_17113片段.xxc'
RE_TRAIN = '--re_train' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
if args:
    TARGET = args[0]

DATA_DIR = r'D:/HDC/data/test'
TRAIN_DIR = r'D:/HDC/data/train'
MODEL_PATH = r'D:/HDC/models/hdc_point_level.pth'
DIMS = 8192
RANDOMNESS_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
THRESHOLD = config.get("test.predict_threshold", 0)
MARGIN = config.get("test.predict_margin", 0)

log("=" * 78)
log(f"randomness 扫描实验   目标文件={TARGET}   re_train={RE_TRAIN}")
log(f"通道={CHANNELS}  NUM_LEVELS={NUM_LEVELS}  DIM={DIMS}")
log(f"判定参数 threshold={THRESHOLD} margin={MARGIN}")
log("=" * 78)


# =========================================================================
# 第一部分（总是执行）：不训练，纯看编码本身的相似度特性
# =========================================================================
def build_level_codebook(r, dims=DIMS, n_levels=NUM_LEVELS, seed=0):
    """按指定 randomness 生成 level 码本，返回 [n_levels, dims]"""
    from torchhd import embeddings as E
    g = torch.Generator().manual_seed(seed)
    emb = E.Level(n_levels, dims, high=256.0, randomness=r, generator=g)
    return emb.weight.detach().clone()


log("")
log("【A】Level 码本相似度特性 vs randomness   (DIM=8192, NUM_LEVELS=256)")
log("-" * 78)
log(f"{'r':>5} | {'相邻':>8} | {'间隔2':>8} | {'间隔5':>8} | {'间隔50':>8} | {'最远(0↔255)':>10} | 单span可分辨level数")
rows_a = []
for r in RANDOMNESS_VALUES:
    W = build_level_codebook(r)
    Wn = torch.nn.functional.normalize(W, dim=1)
    S = (Wn @ Wn.t())

    def avg_interval(k):
        if k >= len(W):
            return float('nan')
        d = torch.diagonal(S, offset=k)
        return float(d.mean())

    adj = avg_interval(1)
    i2 = avg_interval(2)
    i5 = avg_interval(5)
    i50 = avg_interval(50)
    far = float(S[0, -1])
    # levels_per_span = (1-r)*(N-1)+r*1
    lps = (1 - r) * (NUM_LEVELS - 1) + r * 1
    log(f"{r:>5.1f} | {adj:>8.4f} | {i2:>8.4f} | {i5:>8.4f} | {i50:>8.4f} | {far:>10.4f} | {lps:>6.0f}")
    rows_a.append(dict(r=r, adjacent=adj, i2=i2, i5=i5, i50=i50, farthest=far, levels_per_span=lps))

log("")
log("读法：'相邻'应显著低于 1.0 才有分辨力。实测 r=0 时相邻≈0.99 ⇒ 256 个级别几乎不可分。")


# =========================================================================
# 第二部分：加载现有模型，替换编码器，实测真值点上的 max_defect_sim
# =========================================================================
log("")
log("【B】用现有已训练原型 + 各档 randomness 编码，实测真值点的可分性")
log("-" * 78)

hdc_file = f'{DATA_DIR}/{TARGET}_hdc.csv'
label_file = f'{DATA_DIR}/{TARGET}_label.csv'

ds = HDCUnifiedDataset(DATA_DIR, is_train=False, files=[hdc_file], channels=CHANNELS)
log(f"滑窗样本数: {len(ds)}")
if len(ds) == 0:
    log("!! 样本为 0，退出")
    sys.exit(1)

loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

# 真值
df_hdc = pd.read_csv(hdc_file)
df_label = pd.read_csv(label_file)
label_cols = [f'label_{ch}' for ch in CHANNELS]
merged = pd.merge(df_hdc[['encoding']], df_label[['encoding'] + label_cols],
                  on='encoding', how='left').fillna(0)
y_true = merged[label_cols].values

# encoding -> 滑窗样本索引
enc_to_row = {int(e): i for i, e in enumerate(df_hdc['encoding'].values)}
rows = []
for idx, info in enumerate(ds.file_info):
    er = enc_to_row.get(int(info['center_encoding']))
    if er is not None:
        rows.append((er, idx))
pos_rows = [r for r in range(len(y_true)) if (y_true[r] > 0).any()]
log(f"真值点行数: {len(pos_rows)}   能对上滑窗样本的行数: "
    f"{sum(1 for r in pos_rows if any(er == r for er, _ in rows))}")

ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
log(f"模型: dim={ck['dimensions']} channels={ck['channels']}")

log("")
log(f"{'r':>5} | {'判据通过数':>10} | {'真值点max_defect均值':>20} | {'真值点normal均值':>18} | {'最大defect(全样本)':>18}")
log("-" * 78)
rows_b = []
for r in RANDOMNESS_VALUES:
    # 用指定 randomness 重建编码器，但沿用已训练的原型
    model = HDCUnifiedModel(device='cpu', channels=CHANNELS)
    model.dimensions = ck['dimensions']
    model.channels = ck['channels']
    model.num_channels = len(ck['channels'])
    model.is_single_channel = model.num_channels == 1

    enc = HDCPointLevelEncoder(model.dimensions, 128)
    # 只替换 value_hvs 为指定 randomness 的码本，其余保持训练时的随机基
    from torchhd import embeddings as E
    g = torch.Generator().manual_seed(0)
    enc.value_hvs = E.Level(NUM_LEVELS, model.dimensions, high=256.0, randomness=r, generator=g)
    model.encoder = enc
    model.models.load_state_dict(ck['models'])

    with torch.no_grad():
        try:
            preds, normal_sims, max_defect_sims, class_sims = model.predict(
                loader, threshold=THRESHOLD, margin=MARGIN)
        except Exception as e:
            log(f"{r:>5.1f} | 预测失败: {type(e).__name__}: {e}")
            continue

    preds = np.atleast_2d(preds)
    pass_all = int(((max_defect_sims > THRESHOLD) &
                    (max_defect_sims > normal_sims + MARGIN)).sum())

    # 只看真值点所在样本
    idx_map = {er: idx for er, idx in rows}
    sel = [idx_map[r] for r in pos_rows if r in idx_map]
    if sel:
        md_mean = float(max_defect_sims[sel].max(axis=1).mean())
        ns_mean = float(normal_sims[sel].max(axis=1).mean())
    else:
        md_mean = ns_mean = float('nan')

    log(f"{r:>5.1f} | {pass_all:>10d} | {md_mean:>20.4f} | {ns_mean:>18.4f} | "
        f"{float(max_defect_sims.max()):>18.4f}")
    rows_b.append(dict(r=r, pass_all=pass_all, pos_max_defect_mean=md_mean,
                       pos_normal_mean=ns_mean, global_max_defect=float(max_defect_sims.max())))

log("")
log("读法：'判据通过数'=0 说明即便提高 randomness，现有原型也不足以翻正判定；")
log("      '真值点max_defect均值' 若随 r 提升而明显抬升，说明编码分辨力确实是瓶颈之一。")


# =========================================================================
# 第三部分（可选）：每档 randomness 现场重训，看真实指标
# =========================================================================
if RE_TRAIN:
    log("")
    log("【C】每档 randomness 现场重训（data/train），看真实 P/R/F1")
    log("-" * 78)
    import glob
    train_files = sorted(glob.glob(f'{TRAIN_DIR}/*_hdc.csv'))
    log(f"训练文件: {[f.split('/')[-1] for f in train_files]}")
    if len(train_files) == 0:
        log("!! 训练集为空")
    else:
        tr_ds = HDCUnifiedDataset(TRAIN_DIR, is_train=True, files=train_files, channels=CHANNELS)
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=4, shuffle=True)
        log(f"训练样本数: {len(tr_ds)}   epochs={config.get('train.epochs', 1)}  method={config.get('train.method','onlinehd')}")
        log("")
        log(f"{'r':>5} | {'P':>8} | {'R':>8} | {'F1':>8} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TotalPred':>9} | {'Pred=0的比例'}")
        log("-" * 78)
        for r in RANDOMNESS_VALUES:
            try:
                m = HDCUnifiedModel(device='cpu', channels=CHANNELS)
                # 注入指定 randomness 的编码器
                from torchhd import embeddings as E
                enc = HDCPointLevelEncoder(m.dimensions, 128)
                g = torch.Generator().manual_seed(0)
                enc.value_hvs = E.Level(NUM_LEVELS, m.dimensions, high=256.0, randomness=r, generator=g)
                m.encoder = enc
                m.train(tr_loader, epochs=config.get('train.epochs', 1),
                        method=config.get('train.method', 'onlinehd'))
                with torch.no_grad():
                    preds, ns, mds, _ = m.predict(loader, threshold=THRESHOLD, margin=MARGIN)
                preds = np.atleast_2d(preds)
                y_pred = np.zeros_like(y_true)
                for er, idx in rows:
                    y_pred[er] = preds[idx]
                mm = calculate_metrics(y_true, y_pred)
                # 零预测比例
                zero_ratio = float((preds == 0).mean())
                log(f"{r:>5.1f} | {mm['precision']:>8.4f} | {mm['recall']:>8.4f} | {mm['f1']:>8.4f} | "
                    f"{mm['tp']:>5d} | {mm['fp']:>5d} | {mm['fn']:>5d} | {mm['total_pred']:>9d} | {zero_ratio:>6.1%}")
            except Exception as e:
                import traceback
                log(f"{r:>5.1f} | 训练/评估失败: {type(e).__name__}: {e}")
                log(traceback.format_exc())

log("")
log("=" * 78)
log("实验结束（未修改任何模型/数据文件）")
log("=" * 78)

with open('D:/HDC/_sweep_out.txt', 'w', encoding='utf-8') as f:
    f.write(LOG.getvalue())
