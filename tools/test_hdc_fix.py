#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDC 修复回归测试（不依赖 GUI/网络，CPU 可跑）。

覆盖：
1. get_json_files 去重（*.json 与 *.xxc.json 的重叠）
2. _create_encoder 存在 + 分块编码与整窗编码等价
3. 原型不再为零（类均值训练）
4. 单通道形状
5. save/load 往返一致
6. v1 旧检查点被明确拒绝
7. 跨文件预测不串台
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:/HDC")
sys.path.insert(0, r"D:/HDC/torchhd")

from test_hdc_unified import HDCUnifiedDataset, HDCUnifiedModel
from hdc_point_level_classifier import HDCPointLevelEncoder
from hdc_utils import (calculate_metrics, get_json_files,
                       process_prediction_results)

PILOT_TRAIN = r"D:/HDC/output/dataset_prep/pilot_200922_17387"
PILOT_TEST = r"D:/HDC/output/dataset_prep/pilot_210201_18277"
CH = list(range(9))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  OK:", msg)


def main():
    # 1) JSON 去重
    files = get_json_files(Path(PILOT_TRAIN))
    names = [f.name for f in files]
    check(len(names) == len(set(names)) == 1, f"get_json_files 去重: {names}")

    tr = HDCUnifiedDataset(PILOT_TRAIN, is_train=True, channels=CH,
                           train_strategy="all")
    te = HDCUnifiedDataset(PILOT_TEST, is_train=False, channels=CH)
    check(len(tr) > 0 and len(te) > 0, f"数据集加载: train={len(tr)} test={len(te)}")

    model = HDCUnifiedModel(device="cpu", channels=CH, z_threshold=2.75)

    # 2) 编码器 + 分块等价
    enc = model._create_encoder((128, len(CH)))
    x = torch.rand(2, 128, len(CH)) * 100
    enc.chunk_size = 8
    a = enc(x, per_channel=True)
    enc.chunk_size = 128
    b = enc(x, per_channel=True)
    check(tuple(a.shape) == (2, len(CH), model.dimensions), f"per-channel 形状 {tuple(a.shape)}")
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1)).mean().item()
    check(cos > 0.999, f"分块编码与整窗编码等价: cos={cos:.6f}")

    # 3) 训练（小子集）后原型非零
    sub = torch.utils.data.Subset(tr, list(range(0, len(tr), 8)))
    ld = torch.utils.data.DataLoader(sub, batch_size=16, shuffle=True)
    model.train(ld, epochs=1, method="centroid", reset=True)
    check(model.class_counts.sum().item() > 0, "类原型计数非零")
    check(model.defect_mask().sum().item() >= 1, "至少一个通道有伤损原型")
    check(model.score_stats is not None and tuple(model.score_stats.shape) == (len(CH), 2),
          "判据标定形状正确")

    tld = torch.utils.data.DataLoader(te, batch_size=64, shuffle=False)
    p1, ns1, md1, _ = model.predict(tld)
    check(p1.shape == (len(te), len(CH)), f"预测形状 {p1.shape}")

    # 4) 单通道
    tr1 = HDCUnifiedDataset(PILOT_TRAIN, is_train=False, channels=[8])
    check(tr1[0][0].shape == (128, 1), f"单通道样本形状 {tuple(tr1[0][0].shape)}")
    enc1 = HDCPointLevelEncoder(model.dimensions, 128, num_channels=1,
                                num_levels=model.num_levels,
                                input_mode=model.input_mode)
    y = enc1(torch.rand(2, 128, 1) * 100, per_channel=True)
    check(tuple(y.shape) == (2, 1, model.dimensions), f"单通道 HV 形状 {tuple(y.shape)}")

    # 5) save/load 往返
    tmp = Path(tempfile.mkdtemp())
    mp = tmp / "m.pth"
    model.save(mp)
    m2 = HDCUnifiedModel(device="cpu", channels=CH)
    m2.load(mp)
    p2, ns2, md2, _ = m2.predict(tld)
    check(np.allclose(p1, p2) and np.allclose(ns1, ns2),
          "save/load 往返预测一致")
    check(m2.num_levels == model.num_levels and m2.input_mode == model.input_mode,
          "编码配置随模型保存")

    # 6) v1 旧检查点拒绝
    legacy = Path(r"D:/HDC/models/hdc_point_level.pth")
    if legacy.exists():
        ck = torch.load(legacy, map_location="cpu", weights_only=False)
        if int(ck.get("version", 1)) < 2:
            try:
                m2.load(legacy)
                raise AssertionError("v1 检查点未被拒绝")
            except ValueError as e:
                check("旧版检查点" in str(e), "v1 检查点被明确拒绝")
        else:
            print("  SKIP: 主模型已是 v2")

    # 7) 跨文件不串台
    # 两个文件，encoding 相同；只应写入 A 文件的那一条
    file_info = [
        {"file": "A_hdc.csv", "center_encoding": 100, "start": 0},
        {"file": "B_hdc.csv", "center_encoding": 100, "start": 0},
    ]
    import pandas as pd
    df_a = pd.DataFrame({"encoding": [100]})
    preds = np.array([[0], [11]])
    ns = np.array([[0.9], [0.1]])
    md = np.array([[0.1], [0.8]])
    sims = np.zeros((2, 1, 20), dtype=np.float32)
    yp, _, _, _ = process_prediction_results(preds, ns, md, sims,
                                             file_info, df_a, [8],
                                             file_name="A_hdc.csv")
    check(yp[0, 0] == 0, "跨文件串台已修复（A 文件只取自己的预测）")

    print("\nPASS: HDC 修复回归 7 组全部通过")


if __name__ == "__main__":
    main()
