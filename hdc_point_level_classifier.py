#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDC 点级分类算法方案实现
根据 d:\HDC\docs\suanfaliucheng.md#L60-63 算法方案实现
1. 窗长 128, 步长 1
2. 中心行 (index 63) 标签作为窗口结果
3. 9 通道独立模型方案
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from pathlib import Path
from tqdm import tqdm

# 添加 torchhd 库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from torchhd.classifiers import OnlineHD
from torchhd import functional as F

# 配置参数
DIMENSIONS = 8192
NUM_LEVELS = 256
WINDOW_SIZE = 128
CENTER_IDX = 63
NUM_CHANNELS = 9
NUM_TYPES = 20  # 0-19
TOTAL_CLASSES = NUM_TYPES * NUM_CHANNELS  # 180 类别方案

class HDCPointLevelDataset(data.Dataset):
    """
    点级分类 HDC 数据集类
    支持 9 模型方案：每个窗口样本对应中心行 (index 127) 的 9 个通道标签
    """
    def __init__(self, data_dir, is_train=True, window_size=WINDOW_SIZE, step=1, files=None, target_encoding=None):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.window_size = window_size
        self.step = step
        self.samples = []
        self.labels = []
        self.file_info = [] # 存储样本对应的原始信息，用于还原
        self.specified_files = files # 可选：指定要加载的文件列表
        self.target_encoding = target_encoding # 目标点的编码值，用于过滤样本
        self._load_data()

    def _load_data(self):
        """
        加载 CSV 数据并生成滑动窗口样本。
        """
        if self.specified_files is not None:
            csv_files = [Path(f) for f in self.specified_files]
        else:
            csv_files = list(self.data_dir.glob("*_hdc.csv"))
            
        if not csv_files:
            print(f"警告: 在 {self.data_dir} 未找到 *_hdc.csv 文件")
            return

        for csv_file in csv_files:
            # 1. 读取特征数据 (1_hdc.csv)
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"读取文件 {csv_file} 失败: {e}")
                continue

            base_name = csv_file.name.replace("_hdc.csv", "")
            
            # 2. 读取标签数据 (1_label.csv)
            label_file = self.data_dir / f"{base_name}_label.csv"
            if label_file.exists():
                try:
                    label_df = pd.read_csv(label_file)
                    # 确保标签列存在
                    if all(f'label_{i}' in label_df.columns for i in range(NUM_CHANNELS)):
                        # 按照 encoding 合并特征和标签
                        df = pd.merge(df, label_df, on='encoding', how='left').fillna(0)
                    else:
                        # 如果标签文件格式不对，填充 0
                        for i in range(NUM_CHANNELS):
                            df[f'label_{i}'] = 0
                except Exception as e:
                    print(f"读取或合并标签文件 {label_file} 失败: {e}")
                    # 回退到全 0 标签
                    for i in range(NUM_CHANNELS):
                        df[f'label_{i}'] = 0
            else:
                # 如果没有标签文件，所有通道标签设为 0（正常）
                for i in range(NUM_CHANNELS):
                    df[f'label_{i}'] = 0
            
            # 3. 确保所有必要的列都存在
            channel_cols = [f'channel_{i}' for i in range(NUM_CHANNELS)]
            label_cols = [f'label_{i}' for i in range(NUM_CHANNELS)]
            
            if not all(col in df.columns for col in channel_cols):
                print(f"文件 {csv_file.name} 缺少必要的通道列，已跳过")
                continue
            
            if not all(col in df.columns for col in label_cols):
                # 这种情况不应该发生，因为上面已经做了填充，但为了安全再次检查
                for col in label_cols:
                    if col not in df.columns:
                        df[col] = 0
            
            # 按照编码值排序，确保时序/空间连续性
            df = df.sort_values(by='encoding')

            channel_data = df[channel_cols].values.astype(np.float32)
            label_data = df[label_cols].values.astype(np.int64)
            encodings = df['encoding'].values

            # 4. 滑动窗口采样
            if len(df) < self.window_size:
                print(f"跳过文件 {csv_file.name}: 数据行数 ({len(df)}) 小于窗口大小 ({self.window_size})")
                continue

            # 步长为 1，确保每个点都能作为中心点被预测
            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                window_x = channel_data[start:end]
                
                # 获取中心行的 9 个通道标签 (index 63)
                center_labels = label_data[start + CENTER_IDX]
                center_encoding = encodings[start + CENTER_IDX]
                
                # 如果指定了目标编码，只保留包含目标编码的窗口
                if self.target_encoding is not None and center_encoding != self.target_encoding:
                    continue
                
                self.samples.append(window_x)
                self.labels.append(center_labels)
                
                # 无论是否是训练模式，都记录 file_info，但只有在非训练模式下才会用到它进行结果还原
                # 这样可以简化逻辑并避免潜在的索引不匹配问题
                self.file_info.append({
                    'file': str(csv_file),
                    'center_encoding': center_encoding,
                    'start': start
                })
        
        # 如果是训练模式，同步打乱样本和文件信息
        if self.is_train:
            import random
            combined = list(zip(self.samples, self.labels, self.file_info))
            random.shuffle(combined)
            if combined:
                self.samples[:], self.labels[:], self.file_info[:] = zip(*combined)
        
        print(f"数据集加载完成: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.from_numpy(self.samples[idx]), torch.from_numpy(self.labels[idx])

class HDCPointLevelEncoder(nn.Module):
    """
    点级超维编码器
    实现：位置编码(Permutation) + 通道编码(Binding) + 数值编码(Level)
    """
    def __init__(self, out_features, window_size):
        super(HDCPointLevelEncoder, self).__init__()
        self.out_features = out_features
        self.window_size = window_size
        
        # 通道超向量 (9个)
        self.channel_hvs = embeddings.Random(NUM_CHANNELS, out_features)
        # 数值超向量 (256个级别)
        self.value_hvs = embeddings.Level(NUM_LEVELS, out_features, high=256)
        # 位置超向量 (用于绑定)
        self.position_hvs = embeddings.Random(window_size, out_features)

    def forward(self, x):
        """
        前向传播编码逻辑 (向量化加速版)：
        1. 一次性生成所有数值编码 [batch, window_size, channels, DIM]
        2. 批量绑定通道信息
        3. 聚合通道与位置信息
        """
        # x: [batch, window_size, channels]
        batch_size = x.size(0)
        
        # 1. 数值编码: [batch, window_size, channels, out_features]
        # x 形状为 [batch, 128, 9], 映射后约为 72MB (batch=8)，CPU 内存完全可承受
        val_hvs = self.value_hvs(x.long()) 
        
        # 2. 绑定通道信息: [batch, window_size, channels, out_features]
        # channel_hvs: [channels, out_features] -> [1, 1, channels, out_features]
        ch_id_hvs = self.channel_hvs.weight.view(1, 1, NUM_CHANNELS, -1)
        bound_ch = torchhd.bind(val_hvs, ch_id_hvs)
        
        # 3. 聚合通道信息 (Bundling): [batch, window_size, out_features]
        # 对 channels 维度进行求和聚合
        # torchhd.multiset 只支持 dim=-2，这里我们手动求和并保持子类
        time_step_hvs = torch.sum(bound_ch, dim=2).as_subclass(torchhd.MAPTensor)
        
        # 4. 绑定位置信息: [batch, window_size, out_features]
        # position_hvs: [window_size, out_features] -> [1, window_size, out_features]
        pos_hvs = self.position_hvs.weight.unsqueeze(0)
        final_bound = torchhd.bind(time_step_hvs, pos_hvs)
        
        # 5. 聚合整个窗口: [batch, out_features]
        window_hv = torch.sum(final_bound, dim=1).as_subclass(torchhd.MAPTensor)
        
        return torchhd.normalize(window_hv)

class HDCPointLevelModel:
    """
    HDC 点级分类模型管理类 (9 模型方案)
    每个通道拥有独立的分类器 (OnlineHD)
    """
    def __init__(self, dimensions=DIMENSIONS, device='cpu'):
        self.dimensions = dimensions
        self.device = device
        self.encoder = None
        # 使用 OnlineHD 模型，每个通道一个模型
        self.models = nn.ModuleList([Centroid(dimensions, NUM_TYPES) for _ in range(NUM_CHANNELS)]).to(device)

    def train_with_ridge(self, train_loader, alpha=1.0):
        """
        使用岭回归学习方式训练模型
        alpha: 正则化参数，默认为1.0
        """
        print(f"开始训练点级 HDC 模型 (9通道独立模型, 岭回归学习方式, alpha={alpha})...")
        # 初始化编码器
        try:
            first_batch = next(iter(train_loader))
            sample_shape = first_batch[0].shape[1:] # [window_size, channels]
            if self.encoder is None:
                self.encoder = HDCPointLevelEncoder(self.dimensions, sample_shape[0]).to(self.device)
        except StopIteration:
            print("训练集为空，无法训练")
            return

        # 收集所有训练数据
        all_hv = []
        all_targets = []
        
        self.encoder.eval()
        with torch.no_grad():
            with tqdm(train_loader, desc="收集训练数据") as pbar:
                for samples, targets in pbar:
                    samples = samples.to(self.device)
                    targets = targets.to(self.device) # [batch, 9]
                    
                    hv = self.encoder(samples) # [batch, dim]
                    all_hv.append(hv)
                    all_targets.append(targets)
        
        # 合并所有数据
        all_hv = torch.cat(all_hv, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 对每个通道分别执行岭回归学习
        for ch in range(NUM_CHANNELS):
            ch_targets = all_targets[:, ch]
            
            # 转换为one-hot编码
            one_hot = torch.zeros(len(ch_targets), NUM_TYPES, device=self.device)
            one_hot[torch.arange(len(ch_targets)), ch_targets] = 1
            
            # 使用岭回归计算权重
            weights = F.ridge_regression(all_hv, one_hot, alpha=alpha)
            
            # 更新模型权重
            self.models[ch].weight.copy_(weights)
        
        # 所有通道结束后进行归一化，以便后续推理进行相似度计算
        for ch in range(NUM_CHANNELS):
            self.models[ch].normalize()
                
        print("岭回归训练完成。")

    def train(self, train_loader, epochs=10, lr=0.1, method="onlinehd", alpha=1.0):
        """
        训练模型
        method: 学习方式，可选 "onlinehd" 或 "ridge"
        epochs: 迭代轮数 (仅用于 onlinehd)
        lr: 学习率 (仅用于 onlinehd)
        alpha: 正则化参数 (仅用于 ridge)
        """
        if method == "ridge":
            self.train_with_ridge(train_loader, alpha=alpha)
        else:
            # 使用 OnlineHD 学习方式
            print(f"开始训练点级 HDC 模型 (9通道独立模型, Epochs={epochs}, OnlineHD 学习方式)...")
            # 初始化编码器
            try:
                first_batch = next(iter(train_loader))
                sample_shape = first_batch[0].shape[1:] # [window_size, channels]
                if self.encoder is None:
                    self.encoder = HDCPointLevelEncoder(self.dimensions, sample_shape[0]).to(self.device)
            except StopIteration:
                print("训练集为空，无法训练")
                return

            for epoch in range(epochs):
                self.encoder.eval()
                for m in self.models:
                    m.eval()

                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                    for samples, targets in pbar:
                        samples = samples.to(self.device)
                        targets = targets.to(self.device) # [batch, 9]
                        
                        with torch.no_grad():
                            hv = self.encoder(samples) # [batch, dim]
                        
                        # 为每个通道的模型进行 OnlineHD 训练
                        for ch in range(NUM_CHANNELS):
                            ch_targets = targets[:, ch]
                            # 使用 add_online 方法：只在预测错误时更新原型向量
                            self.models[ch].add_online(hv, ch_targets, lr=lr)
            
            # 所有轮次结束后进行归一化，以便后续推理进行相似度计算
            for ch in range(NUM_CHANNELS):
                self.models[ch].normalize()
                    
            print("训练完成。")

    def predict(self, test_loader, threshold=0.1, margin=0.0):
        """
        预测类别，基于类别自身阈值判断
        threshold: 最小相似度要求
        margin: 保留参数，不再使用
        返回: (predictions, normal_sims, max_defect_sims, all_class_sims)
        """
        print(f"开始预测 (9通道并行预测, Threshold={threshold})...")
        if self.encoder is None:
            print("错误: 编码器未初始化")
            return [], [], [], []
            
        self.encoder.eval()
        for m in self.models:
            m.eval()
        
        all_preds = []
        all_normal_sims = []
        all_max_defect_sims = []
        all_class_sims = []  # 存储所有类别的相似度
        
        with torch.no_grad():
            for samples, _ in tqdm(test_loader, desc="预测中"):
                samples = samples.to(self.device)
                hv = self.encoder(samples) # [batch, dim]
                
                batch_preds = []
                batch_normal_sims = []
                batch_max_defect_sims = []
                batch_class_sims = []  # 存储当前批次所有类别的相似度
                
                for ch in range(NUM_CHANNELS):
                    # 获取所有类别的余弦相似度 (范围 -1 到 1)
                    output = self.models[ch](hv) # dot=False, 计算余弦相似度
                    
                    # 获取正常类(0)的相似度
                    normal_sim = output[:, 0]
                    
                    # 获取除正常类外最相似的伤损类
                    defect_output = output[:, 1:] # 排除索引0
                    max_defect_sim, defect_pred = torch.max(defect_output, dim=1)
                    defect_pred = defect_pred + 1 # 还原索引
                    
                    # 判定逻辑：
                    # 伤损类相似度必须大于基础阈值
                    is_defect = (max_defect_sim > threshold)
                    
                    final_pred = torch.zeros_like(normal_sim, dtype=torch.long)
                    final_pred[is_defect] = defect_pred[is_defect]
                    
                    batch_preds.append(final_pred.unsqueeze(1))
                    batch_normal_sims.append(normal_sim.unsqueeze(1))
                    batch_max_defect_sims.append(max_defect_sim.unsqueeze(1))
                    batch_class_sims.append(output.unsqueeze(1))  # 存储所有类别的相似度
                
                # 合并 9 个通道的结果
                combined_pred = torch.cat(batch_preds, dim=1)
                combined_normal_sim = torch.cat(batch_normal_sims, dim=1)
                combined_max_defect_sim = torch.cat(batch_max_defect_sims, dim=1)
                combined_class_sims = torch.cat(batch_class_sims, dim=1)  # 合并所有类别的相似度
                
                all_preds.append(combined_pred.cpu())
                all_normal_sims.append(combined_normal_sim.cpu())
                all_max_defect_sims.append(combined_max_defect_sim.cpu())
                all_class_sims.append(combined_class_sims.cpu())  # 添加到总列表
        
        return (
            torch.cat(all_preds, dim=0).numpy(), 
            torch.cat(all_normal_sims, dim=0).numpy(), 
            torch.cat(all_max_defect_sims, dim=0).numpy(),
            torch.cat(all_class_sims, dim=0).numpy()  # 返回所有类别的相似度
        )

    def save(self, path):
        state = {
            'encoder': self.encoder.state_dict() if self.encoder else None,
            'models': self.models.state_dict(),
            'dimensions': self.dimensions
        }
        torch.save(state, path)
        print(f"模型已保存至: {path}")

    def load(self, path, window_size=WINDOW_SIZE):
        # 使用 weights_only=False 以兼容自定义类 (如 torchhd 的 Tensor 类型)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # 兼容旧版本 PyTorch 不支持 weights_only 参数
            checkpoint = torch.load(path, map_location=self.device)
            
        self.dimensions = checkpoint['dimensions']
        if self.dimensions != DIMENSIONS:
            print(f"[WARN] 模型维度不匹配 (模型: {self.dimensions}, 配置: {DIMENSIONS})，将重新初始化。")
            self.dimensions = DIMENSIONS
            self.encoder = HDCPointLevelEncoder(self.dimensions, window_size).to(self.device)
            return

        self.encoder = HDCPointLevelEncoder(self.dimensions, window_size).to(self.device)
        if checkpoint['encoder']:
            self.encoder.load_state_dict(checkpoint['encoder'])
        self.models.load_state_dict(checkpoint['models'])
        print(f"模型已从 {path} 加载。")

def main():
    # 示例用法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    train_dir = r"d:\HDC\data\train"
    test_dir = r"d:\HDC\data\test"
    model_path = r"d:\HDC\models\hdc_point_level_9ch.pth"
    
    # 创建模型保存目录
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 1. 训练
    if os.path.exists(train_dir):
        train_ds = HDCPointLevelDataset(train_dir, is_train=True)
        if len(train_ds) > 0:
            # 显存优化：减小 batch_size
            train_ld = data.DataLoader(train_ds, batch_size=8, shuffle=True)
            hdc_sys = HDCPointLevelModel(device=device)
            hdc_sys.train(train_ld)
            hdc_sys.save(model_path)
        else:
            print("训练集样本数为 0")
    
    # 2. 预测
    if os.path.exists(test_dir):
        test_ds = HDCPointLevelDataset(test_dir, is_train=False)
        if len(test_ds) > 0:
            # 显存优化：测试 Batch Size 也减小
            test_ld = data.DataLoader(test_ds, batch_size=8, shuffle=False)
            
            hdc_sys = HDCPointLevelModel(device=device)
            if os.path.exists(model_path):
                hdc_sys.load(model_path)
                predictions, normal_sims, max_defect_sims, all_class_sims = hdc_sys.predict(test_ld) # [N, 9]
                
                # 3. 处理预测结果并生成 1_predicted_label.csv
                print("处理预测结果...")
                file_results = {}
                for idx, pred_row in enumerate(predictions):
                    info = test_ds.file_info[idx]
                    file_name = Path(info['file']).name
                    if file_name not in file_results:
                        file_results[file_name] = []
                    
                    file_results[file_name].append({
                        'encoding': info['center_encoding'],
                        'labels': pred_row, # 9 个通道的预测值
                        'normal_sims': normal_sims[idx], # 9 个通道的正常类相似度
                        'max_defect_sims': max_defect_sims[idx] # 9 个通道的最强伤损相似度
                    })
                    
                for file_name, results in file_results.items():
                    # 读取原始数据以获取所有 encoding
                    df_orig = pd.read_csv(Path(test_dir) / file_name)
                    pred_df = pd.DataFrame({'encoding': df_orig['encoding']})
                    for i in range(NUM_CHANNELS):
                        pred_df[f'predicted_label_{i}'] = 0
                        pred_df[f'similarity_{i}'] = "0.0,0.0"
                    
                    # 填充预测值
                    results_data = []
                    for r in results:
                        row = {'encoding': r['encoding']}
                        for i in range(NUM_CHANNELS):
                            row[f'predicted_label_{i}'] = r['labels'][i]
                            # 格式化相似度为 "正常,伤损"
                            sim_str = f"{r['normal_sims'][i]:.4f},{r['max_defect_sims'][i]:.4f}"
                            row[f'similarity_{i}'] = sim_str
                        results_data.append(row)
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # 合并结果
                    pred_df = pd.merge(pred_df[['encoding']], results_df, on='encoding', how='left').fillna(0)
                    
                    out_name = file_name.replace("_hdc.csv", "_predicted_label.csv")
                    pred_df.to_csv(Path(test_dir) / out_name, index=False)
                    print(f"预测结果已保存至: {out_name}")
            else:
                print(f"未找到模型文件: {model_path}")
        else:
            print("测试集样本数为 0")

if __name__ == "__main__":
    main()
