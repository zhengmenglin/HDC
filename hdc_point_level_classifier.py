#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDC 点级分类算法实现（2026-10-21 修复版）

设计要点（与旧版的差异都写在这里，改代码前先读）：
1. 窗长 128，中心点 index 63 的 9 通道标签作为该窗口的监督信号；
2. 每个通道独立一个 Centroid 分类器（不再共用 9 个固定槽位，支持通道子集）；
3. 编码 = Level(值) ⊕ bind(Random(通道)) ⊕ bind(Random(位置)) → sum → normalize；
4. **原型用有监督类均值累加**，不再依赖 Centroid 的零初始化 + add_online——
   后者在“全预测为 0 且标签也是 0”时会自锁，原型永远为零（旧版 P=R=F1=0 的根因之一）；
5. **相似度判据用训练集标定的 z-score**，不再用裸 cosine margin——
   旧版 normal_sim≈0.918 / max_defect≈0.184 时，`md > ns + margin` 恒假；
6. 输入默认做“窗内 z-score”再量化到 Level：
   超声信号有很强的公共分量（尤其通道 8 底波），直接量化会让所有窗口的 HV 几乎相同；
   窗内标准化把“相对起伏”放大，是让编码可分的最小改动。
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from pathlib import Path
from tqdm import tqdm

# 添加 torchhd 库路径
sys.path.append(os.path.join(os.path.dirname(__file__), "torchhd"))
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

# ---------------- 配置参数 ----------------
DIMENSIONS = 8192
NUM_LEVELS = 64          # 旧版 256 超出 8192 维的分辨力（经验 ~O(√D)≈90）；64 留安全余量
WINDOW_SIZE = 128
CENTER_IDX = 63
NUM_CHANNELS = 9
NUM_TYPES = 20           # 0-19
TOTAL_CLASSES = NUM_TYPES * NUM_CHANNELS  # 兼容旧引用

INPUT_MODE = os.environ.get("HDC_INPUT_MODE", "window_z")  # raw | window_z
CLIP_Z = 3.0             # 窗内 z-score 的截断范围，映射到 [0, NUM_LEVELS-1]
VERBOSE = os.environ.get("HDC_VERBOSE", "0") == "1"


def _log(msg):
    if VERBOSE:
        print(msg)


class HDCPointLevelDataset(data.Dataset):
    """点级分类数据集：滑动窗口，中心行标签作为监督。

    与旧版的区别：支持 channels 子集，标签文件按 CSV 所在目录查找
    （旧版固定用 data_dir，指定外部文件时会静默读不到标签）。
    """

    def __init__(self, data_dir, is_train=True, window_size=WINDOW_SIZE, step=1,
                 files=None, target_encoding=None, channels=None, seed=42):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.window_size = window_size
        self.step = step
        self.samples = []
        self.labels = []
        self.file_info = []
        self.specified_files = files
        self.target_encoding = target_encoding
        self.channel_ids = list(channels) if channels is not None else list(range(NUM_CHANNELS))
        self.channel_cols = [f"channel_{c}" for c in self.channel_ids]
        self.label_cols = [f"label_{c}" for c in self.channel_ids]
        self.seed = seed
        self._load_data()

    def _load_data(self):
        if self.specified_files is not None:
            csv_files = [Path(f) for f in self.specified_files]
        else:
            csv_files = sorted(self.data_dir.glob("*_hdc.csv"))

        if not csv_files:
            print(f"警告: 在 {self.data_dir} 未找到 *_hdc.csv 文件")
            return

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"读取文件 {csv_file} 失败: {e}")
                continue

            base_name = csv_file.name.replace("_hdc.csv", "")
            # 标签必须跟着 CSV 走，而不是跟着 data_dir 走
            label_file = csv_file.parent / f"{base_name}_label.csv"
            if label_file.exists():
                try:
                    label_df = pd.read_csv(label_file)
                    merge_cols = ["encoding"] + [c for c in self.label_cols if c in label_df.columns]
                    if len(merge_cols) > 1:
                        df = pd.merge(df, label_df[merge_cols], on="encoding", how="left").fillna(0)
                    for c in self.label_cols:
                        if c not in df.columns:
                            df[c] = 0
                except Exception as e:
                    print(f"读取或合并标签文件 {label_file} 失败: {e}")
                    for c in self.label_cols:
                        df[c] = 0
            else:
                for c in self.label_cols:
                    df[c] = 0

            if not all(col in df.columns for col in self.channel_cols):
                print(f"文件 {csv_file.name} 缺少通道列 {self.channel_cols}，已跳过")
                continue

            df = df.sort_values(by="encoding")
            channel_data = df[self.channel_cols].values.astype(np.float32)
            label_data = df[self.label_cols].values.astype(np.int64)
            encodings = df["encoding"].values

            if len(df) < self.window_size:
                print(f"跳过文件 {csv_file.name}: 数据行数 ({len(df)}) 小于窗口大小 ({self.window_size})")
                continue

            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                window_x = channel_data[start:end]
                center_labels = label_data[start + CENTER_IDX]
                center_encoding = encodings[start + CENTER_IDX]

                if self.target_encoding is not None and center_encoding != self.target_encoding:
                    continue

                self.samples.append(window_x)
                self.labels.append(center_labels)
                self.file_info.append({
                    "file": str(csv_file),
                    "center_encoding": center_encoding,
                    "start": start,
                })

        if self.is_train and self.samples:
            # 固定种子，避免每次训练样本顺序不同导致结果漂移
            import random
            rng = random.Random(self.seed)
            combined = list(zip(self.samples, self.labels, self.file_info))
            rng.shuffle(combined)
            self.samples, self.labels, self.file_info = map(list, zip(*combined))

        print(f"数据集加载完成: {len(self.samples)} 个样本 "
              f"({len(self.channel_ids)} 通道: {self.channel_ids})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.from_numpy(self.labels[idx])


class HDCPointLevelEncoder(nn.Module):
    """点级超维编码器：Level(值) ⊕ bind(通道) ⊕ bind(位置) → sum → normalize。"""

    def __init__(self, out_features, window_size, num_channels=NUM_CHANNELS,
                 num_levels=NUM_LEVELS, input_mode=INPUT_MODE, clip_z=CLIP_Z):
        super().__init__()
        self.out_features = out_features
        self.window_size = window_size
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.input_mode = input_mode
        self.clip_z = clip_z
        # 按位置分块编码：整窗一次物化 [B,128,C,8192] 在 CUDA batch=128 时
        # 要 4.5GB/张量，必然 OOM；分块后峰值降到 ~1/16。
        self.chunk_size = 8

        self.channel_hvs = embeddings.Random(num_channels, out_features)
        # Level 的 high 用 num_levels，输入按下标 0..num_levels-1 取级
        self.value_hvs = embeddings.Level(num_levels, out_features, high=num_levels)
        self.position_hvs = embeddings.Random(window_size, out_features)

        if VERBOSE:
            self._validate_hypervectors()

    def _to_level_index(self, x):
        """把原始幅值映射到 [0, num_levels-1]。

        window_z：每个窗口、每个通道各自标准化。目的不是“提特征”，
        而是去掉窗口内的直流/公共分量——否则 128 个位置绑定后，
        所有样本的 HV 都指向同一方向（旧版 normal_sim≈0.918 的来源）。
        """
        if self.input_mode == "raw":
            return x.round().long().clamp_(0, self.num_levels - 1)

        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
        z = (x - mu) / sd
        z = z.clamp(-self.clip_z, self.clip_z)
        idx = ((z + self.clip_z) / (2.0 * self.clip_z) * (self.num_levels - 1)).round().long()
        return idx.clamp_(0, self.num_levels - 1)

    def forward(self, x, per_channel=False):
        """编码一个窗口。

        per_channel=True 返回 [B, C, D]：每个通道自己的窗口 HV。
        这是“9 通道独立建模”的关键——旧版把所有通道 sum 成一个 HV，
        再让 9 个分类器共用，通道越多彼此稀释越严重（实测 9 通道 F1 从
        0.58 掉到 0.24）。per_channel=False 仅保留给旧的单 HV 调用。
        """
        idx = self._to_level_index(x)
        B, W, C = idx.shape
        ch_id_hvs = self.channel_hvs.weight.view(1, 1, self.num_channels, -1)

        acc = None
        for t0 in range(0, W, self.chunk_size):
            t1 = min(W, t0 + self.chunk_size)
            val_hvs = self.value_hvs(idx[:, t0:t1, :])          # [B, chunk, C, D]
            bound = torchhd.bind(val_hvs, ch_id_hvs)            # 绑通道
            pos_hvs = self.position_hvs.weight[t0:t1].view(1, t1 - t0, 1, -1)
            bound = torchhd.bind(bound, pos_hvs)                # 绑位置
            step = torch.sum(bound, dim=1)                      # [B, C, D]
            acc = step if acc is None else acc + step

        per = torchhd.normalize(acc.as_subclass(torchhd.MAPTensor))  # [B, C, D]
        if per_channel:
            return per

        window_hv = torch.sum(per, dim=1).as_subclass(torchhd.MAPTensor)
        return torchhd.normalize(window_hv)

    def _validate_hypervectors(self):
        print("\n[HDC] 验证超向量相似度分布...")
        self._check_similarity(self.channel_hvs.weight, "通道")
        self._check_similarity(self.position_hvs.weight, "位置")
        try:
            self._check_level_similarity(self.value_hvs.weight, "数值")
        except AttributeError:
            print("[INFO] 数值超向量使用 Level 编码，无需验证相似度")

    @staticmethod
    def _check_similarity(hvs, name):
        norm_hvs = torch.nn.functional.normalize(hvs, dim=1)
        sim_matrix = torch.matmul(norm_hvs, norm_hvs.t())
        triu = torch.triu(sim_matrix, diagonal=1)
        vals = triu[triu > 0]
        if len(vals):
            print(f"{name} 超向量相似度: mean={vals.mean():.4f} max={vals.max():.4f} "
                  f"min={vals.min():.4f} std={vals.std():.4f}")

    @staticmethod
    def _check_level_similarity(hvs, name):
        norm_hvs = torch.nn.functional.normalize(hvs, dim=1)
        sim_matrix = torch.matmul(norm_hvs, norm_hvs.t())
        adj = [sim_matrix[i, i + 1].item() for i in range(len(hvs) - 1)]
        if adj:
            print(f"{name} Level 相邻相似度: {sum(adj) / len(adj):.4f}")
        if len(hvs) >= 10:
            print(f"{name} Level 最远相似度: {sim_matrix[0, -1].item():.4f}")


class HDCPointLevelModel:
    """9 通道（或通道子集）独立的 HDC 分类器。

    原型保存的是“类内样本 HV 之和”，不是均值；cosine 对尺度不变，
    这样增量训练可以直接累加，不需要反解均值。
    """

    def __init__(self, dimensions=DIMENSIONS, device="cpu", channels=None,
                 num_levels=NUM_LEVELS, input_mode=INPUT_MODE, z_threshold=2.75):
        self.dimensions = dimensions
        self.device = device
        self.channels = list(channels) if channels is not None else list(range(NUM_CHANNELS))
        self.num_channels = len(self.channels)
        self.num_levels = num_levels
        self.input_mode = input_mode
        self.z_threshold = z_threshold
        self.window_size = WINDOW_SIZE
        self.encoder = None
        self.models = nn.ModuleList([Centroid(dimensions, NUM_TYPES) for _ in self.channels]).to(device)
        # 类计数用于增量训练/诊断；CPU float64，避免长期累加误差
        self.class_counts = torch.zeros(self.num_channels, NUM_TYPES, dtype=torch.float64)
        self.score_stats = None  # [num_channels, 2]：(mu, sigma) of (max_defect_sim - normal_sim)

    # ---------------- 编码器 ----------------
    def _create_encoder(self, sample_shape):
        """旧版缺失的方法（test_hdc_unified.py:348/486 调用）——补上。"""
        window_size = int(sample_shape[0])
        return HDCPointLevelEncoder(
            self.dimensions, window_size,
            num_channels=self.num_channels,
            num_levels=self.num_levels,
            input_mode=self.input_mode,
        ).to(self.device)

    def _ensure_encoder(self, loader):
        if self.encoder is not None:
            return True
        try:
            samples, _ = next(iter(loader))
        except StopIteration:
            print("训练集为空，无法初始化编码器")
            return False
        self.window_size = int(samples.shape[1])
        self.encoder = self._create_encoder(samples.shape[1:])
        return True

    def _cosine_scores(self, hv, ch_idx):
        """手工计算余弦相似度。

        不用 Centroid.forward 的原因：没有正样本的类原型是零向量，
        torchhd 会走 0/0 并打 "norm nearly zero" 警告；这里把零向量
        的分数定义为 0，既无警告也不会污染 argmax。
        """
        w = self.models[ch_idx].weight.detach().float()
        h = hv.detach().float()
        wn = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
        hn = h.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.matmul(h, (w / wn).t()) / hn

    def defect_mask(self):
        """哪些通道在训练集里见过伤损类（class 1..19）。"""
        return (self.class_counts[:, 1:].sum(dim=1) > 0)

    # ---------------- 训练 ----------------
    def fit_centroids(self, train_loader, reset=True):
        """有监督类均值累加：一次遍历即可，不会出现零原型死锁。

        reset=False 时在现有原型上累加（增量学习）；class_counts 同步累加。
        """
        if not self._ensure_encoder(train_loader):
            return
        if reset:
            for m in self.models:
                m.weight.zero_()
            self.class_counts.zero_()
        # class_counts 固定放 CPU：它只做计数，避免 map_location=cuda 把它搬走后
        # 与 torch.bincount 的 CPU 结果设备不一致。
        self.class_counts = self.class_counts.cpu()

        self.encoder.eval()
        for m in self.models:
            m.eval()

        with torch.no_grad():
            for samples, targets in tqdm(train_loader, desc="拟合类原型"):
                samples = samples.to(self.device)
                targets = targets.to(self.device).long()
                hv = self.encoder(samples, per_channel=True)  # [B, C, D]
                for i in range(self.num_channels):
                    t = targets[:, i]
                    self.models[i].weight.index_add_(0, t, hv[:, i])
                    self.class_counts[i] += torch.bincount(
                        t.detach().cpu(), minlength=NUM_TYPES).to(self.class_counts.dtype)

        nonzero = int((self.class_counts > 0).sum().item())
        print(f"类原型拟合完成：{nonzero}/{self.num_channels * NUM_TYPES} 个类非零")

    def calibrate(self, loader):
        """用训练集的正常样本标定每个通道的 (md - ns) 分布。

        旧版用裸 cosine margin，在两类原型质量不对称时恒假；
        这里把判据换成 z-score：伤损分数必须显著高于“正常样本的常规波动”。
        """
        if self.encoder is None:
            return
        diffs = [[] for _ in range(self.num_channels)]
        self.encoder.eval()
        for m in self.models:
            m.eval()
        with torch.no_grad():
            for samples, targets in loader:
                samples = samples.to(self.device)
                targets = targets.to(self.device).long()
                hv = self.encoder(samples, per_channel=True)
                for i in range(self.num_channels):
                    out = self._cosine_scores(hv[:, i], i)
                    ns = out[:, 0]
                    md = out[:, 1:].max(dim=1).values
                    if not bool(self.defect_mask()[i]):
                        md = torch.zeros_like(md)
                    normal = targets[:, i] == 0
                    if normal.any():
                        diffs[i].append((md - ns)[normal].cpu())

        stats = []
        for i in range(self.num_channels):
            if diffs[i]:
                d = torch.cat(diffs[i]).double()
                sd_raw = d.std(unbiased=False).item()
                if len(d) >= 10 and sd_raw >= 1e-4:
                    mu = d.mean().item()
                    sd = sd_raw
                else:
                    # 分数几乎没有波动（典型：ridge 权重退化）⇒ 标定失效。
                    # 退回 (0,1)，让 z≈原始分，避免 sigma→0 把噪声放大成海量虚警。
                    mu, sd = 0.0, 1.0
            else:
                mu, sd = 0.0, 1.0
            stats.append((mu, sd))
        self.score_stats = torch.tensor(stats, dtype=torch.float32)
        print("判据标定完成（每通道 mu/sigma）:")
        for i, (mu, sd) in enumerate(stats):
            print(f"  ch{self.channels[i]}: mu={mu:+.4f} sigma={sd:.4f}")

    def train_with_ridge(self, train_loader, alpha=1.0, reset=True):
        print(f"开始训练点级 HDC 模型（{self.num_channels} 通道，岭回归，alpha={alpha}）...")
        if not self._ensure_encoder(train_loader):
            return

        all_hv, all_targets = [], []
        self.encoder.eval()
        with torch.no_grad():
            for samples, targets in tqdm(train_loader, desc="收集训练数据"):
                samples = samples.to(self.device)
                hv = self.encoder(samples, per_channel=True)
                all_hv.append(hv)
                all_targets.append(targets.to(self.device))

        all_hv = torch.cat(all_hv, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 岭回归不经过 fit_centroids，必须同步类计数，否则 defect_mask 全 False
        # （所有通道都会被当成“没学过伤损”，预测恒为 0）。
        self.class_counts.zero_()
        for i in range(self.num_channels):
            self.class_counts[i] += torch.bincount(
                all_targets[:, i].long().detach().cpu(), minlength=NUM_TYPES).double()

        for i in range(self.num_channels):
            ch_targets = all_targets[:, i].long()
            one_hot = torch.zeros(len(ch_targets), NUM_TYPES, device=self.device)
            one_hot[torch.arange(len(ch_targets)), ch_targets] = 1

            # 岭回归的对偶形式：样本数 n 通常远小于维度 d（这里是 232 vs 8192）。
            # torchhd.functional.ridge_regression 走 d×d 的 pinv，GPU 上会卡死；
            # 用 Woodbury 恒等式解 n×n 系统，等价且快几个数量级。
            X = all_hv[:, i].float()
            n, d = X.shape
            if n <= d:
                A = X @ X.t() + alpha * torch.eye(n, device=X.device)
                W = X.t() @ torch.linalg.solve(A, one_hot)
            else:
                A = X.t() @ X + alpha * torch.eye(d, device=X.device)
                W = torch.linalg.solve(A, X.t() @ one_hot)
            self.models[i].weight.copy_(W.t())

        self.calibrate(train_loader)
        print("岭回归训练完成。")

    def train(self, train_loader, epochs=1, lr=0.1, method="centroid", alpha=1.0, reset=True):
        """训练入口。

        method:
          - centroid（默认）：有监督类均值，稳定、无死锁；
          - onlinehd：先类均值，再用 OnlineHD 在训练集上做少量纠错；
          - ridge：岭回归（保留旧能力）。
        reset=False 用于增量学习：在已有原型上累加。
        """
        if method == "ridge":
            self.train_with_ridge(train_loader, alpha=alpha, reset=reset)
            return

        self.fit_centroids(train_loader, reset=reset)

        if method == "onlinehd" and epochs > 0:
            print(f"OnlineHD 纠错：{epochs} 个 epoch，lr={lr}")
            if not self._ensure_encoder(train_loader):
                return
            for m in self.models:
                m.normalize()
            self.encoder.eval()
            for epoch in range(epochs):
                for samples, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                    samples = samples.to(self.device)
                    targets = targets.to(self.device).long()
                    with torch.no_grad():
                        hv = self.encoder(samples, per_channel=True)
                        for i in range(self.num_channels):
                            self.models[i].add_online(hv[:, i], targets[:, i], lr=lr)
            for m in self.models:
                m.normalize()

        self.calibrate(train_loader)
        print("训练完成。")

    # ---------------- 推理 ----------------
    def predict(self, test_loader, threshold=None, margin=None, z_threshold=None):
        """返回 (predictions, normal_sims, max_defect_sims, all_class_sims)。

        有 score_stats 时走 z-score 判据：z = (md - ns - mu) / sigma > z_threshold；
        否则回退到旧版的 raw 判据（md > threshold 且 md > ns + margin），仅用于兼容。
        """
        if self.encoder is None:
            print("错误: 编码器未初始化")
            return [], [], [], []

        self.encoder.eval()
        for m in self.models:
            m.eval()

        all_preds, all_ns, all_md, all_sims = [], [], [], []
        use_z = self.score_stats is not None
        z_thr = self.z_threshold if z_threshold is None else z_threshold
        print(f"开始预测（{self.num_channels} 通道，"
              f"{'z-score 阈值=' + str(z_thr) if use_z else 'raw 判据'}）...")

        with torch.no_grad():
            for samples, _ in tqdm(test_loader, desc="预测中"):
                samples = samples.to(self.device)
                hv = self.encoder(samples, per_channel=True)

                batch_preds, batch_ns, batch_md, batch_sims = [], [], [], []
                dmask = self.defect_mask()
                for i in range(self.num_channels):
                    out = self._cosine_scores(hv[:, i], i)  # cosine similarity
                    ns = out[:, 0]
                    md, dp = out[:, 1:].max(dim=1)
                    dp = dp + 1

                    if not bool(dmask[i]):
                        # 该通道训练集没有伤损样本：只学正常原型，不应报伤损
                        is_defect = torch.zeros_like(ns, dtype=torch.bool)
                    elif use_z:
                        mu = self.score_stats[i, 0].to(out.device)
                        sd = self.score_stats[i, 1].to(out.device)
                        z = (md - ns - mu) / sd
                        is_defect = z > z_thr
                    else:
                        t = 0.1 if threshold is None else threshold
                        mg = 0.0 if margin is None else margin
                        is_defect = (md > t) & (md > ns + mg)

                    pred = torch.zeros_like(ns, dtype=torch.long)
                    pred[is_defect] = dp[is_defect]

                    batch_preds.append(pred.unsqueeze(1))
                    batch_ns.append(ns.unsqueeze(1))
                    batch_md.append(md.unsqueeze(1))
                    batch_sims.append(out.unsqueeze(1))

                all_preds.append(torch.cat(batch_preds, dim=1).cpu())
                all_ns.append(torch.cat(batch_ns, dim=1).cpu())
                all_md.append(torch.cat(batch_md, dim=1).cpu())
                all_sims.append(torch.cat(batch_sims, dim=1).cpu())

        return (
            torch.cat(all_preds, dim=0).numpy(),
            torch.cat(all_ns, dim=0).numpy(),
            torch.cat(all_md, dim=0).numpy(),
            torch.cat(all_sims, dim=0).numpy(),
        )

    # ---------------- 持久化 ----------------
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 2,
            "dimensions": self.dimensions,
            "channels": self.channels,
            "num_levels": self.num_levels,
            "window_size": self.window_size,
            "input_mode": self.input_mode,
            "encoder": self.encoder.state_dict() if self.encoder else None,
            "models": self.models.state_dict(),
            "class_counts": self.class_counts,
            "score_stats": self.score_stats,
        }
        torch.save(state, path)
        print(f"模型已保存至: {path}")

    def load(self, path, window_size=WINDOW_SIZE, **_ignored):
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location=self.device)

        # v1 = 旧版（raw 输入 + 256 级 Level + 零初始化原型）。它的原型本身
        # 就是坏的，且编码语义与当前 window_z/64 级不兼容；静默加载会得到
        # “看似能跑、结果全错”的模型，必须明确拒绝。
        if int(ck.get("version", 1)) < 2:
            raise ValueError(
                "旧版检查点（v1：raw 编码 / 256 级 / 零原型）与当前编码不兼容，"
                "请用 --force-retrain 从零训练。"
            )

        self.dimensions = int(ck.get("dimensions", DIMENSIONS))
        self.channels = list(ck.get("channels", list(range(NUM_CHANNELS))))
        self.num_channels = len(self.channels)
        self.num_levels = int(ck.get("num_levels", NUM_LEVELS))
        self.window_size = int(ck.get("window_size", window_size))
        self.input_mode = ck.get("input_mode", INPUT_MODE)

        enc_state = ck.get("encoder")
        if not enc_state:
            raise ValueError(
                "模型文件缺少 encoder（旧版只保存了原型）。"
                "编码与原型必须同源，不能只加载原型——请用 --force-retrain 从零训练。"
            )

        self.encoder = self._create_encoder((self.window_size, self.num_channels))
        self.encoder.load_state_dict(enc_state)

        # 维度/通道数可能与当前配置不同，重建分类器
        self.models = nn.ModuleList(
            [Centroid(self.dimensions, NUM_TYPES) for _ in self.channels]).to(self.device)
        self.models.load_state_dict(ck["models"])

        cc = ck.get("class_counts")
        if cc is not None and tuple(cc.shape) == (self.num_channels, NUM_TYPES):
            self.class_counts = cc.double().cpu()
        else:
            self.class_counts = torch.zeros(self.num_channels, NUM_TYPES, dtype=torch.float64)

        ss = ck.get("score_stats")
        if ss is not None and tuple(ss.shape) == (self.num_channels, 2):
            self.score_stats = ss.float().cpu()
        else:
            self.score_stats = None

        print(f"模型已从 {path} 加载（{self.num_channels} 通道，D={self.dimensions}，"
              f"levels={self.num_levels}，mode={self.input_mode}）")


def main():
    """最小自检：训练 + 预测，供手工排障用。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(r"D:/HDC/data/train")
    test_dir = Path(r"D:/HDC/data/test")
    model_path = Path(r"D:/HDC/models/hdc_point_level_9ch.pth")

    hdc_sys = HDCPointLevelModel(device=device)
    if train_dir.exists():
        ds = HDCPointLevelDataset(train_dir, is_train=True)
        if len(ds):
            ld = data.DataLoader(ds, batch_size=8, shuffle=True)
            hdc_sys.train(ld, epochs=1, method="centroid")
            hdc_sys.save(model_path)

    if test_dir.exists() and model_path.exists():
        ds = HDCPointLevelDataset(test_dir, is_train=False)
        if len(ds):
            ld = data.DataLoader(ds, batch_size=8, shuffle=False)
            hdc_sys.load(model_path)
            preds, ns, md, sims = hdc_sys.predict(ld)
            print("预测完成:", preds.shape, ns.shape, md.shape, sims.shape)


if __name__ == "__main__":
    main()
