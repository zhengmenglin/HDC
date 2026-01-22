#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDC 点级分类算法方案实现
根据 d:\\HDC\\docs\\suanfaliucheng.md#L60-63 算法方案实现
1. 窗长 256, 步长 1
2. 中心行 (index 127) 标签作为窗口结果
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

# 配置参数
DIMENSIONS = 10000
NUM_LEVELS = 256
WINDOW_SIZE = 256
CENTER_IDX = 127
NUM_CHANNELS = 9
NUM_TYPES = 20  # 0-19
TOTAL_CLASSES = NUM_TYPES * NUM_CHANNELS  # 180 类别方案

class HDCPointLevelDataset(data.Dataset):
    """
    点级分类 HDC 数据集类
    支持 9 模型方案：每个窗口样本对应中心行 (index 127) 的 9 个通道标签
    """
    def __init__(self, data_dir, is_train=True, window_size=WINDOW_SIZE, step=1):
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.window_size = window_size
        self.step = step
        self.samples = []
        self.labels = [] # 形状为 [N, 9]
        self.file_info = []
        
        self._load_data()

    def _load_data(self):
        """
        加载 CSV 数据并生成滑动窗口样本。
        """
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
                    # 按照 encoding 合并特征和标签
                    df = pd.merge(df, label_df, on='encoding', how='left').fillna(0)
                except Exception as e:
                    print(f"读取或合并标签文件 {label_file} 失败: {e}")
                    # 回退到全 0 标签
                    for i in range(NUM_CHANNELS):
                        df[f'label_{i}'] = 0
            else:
                # 如果没有标签文件，所有通道标签设为 0（正常）
                for i in range(NUM_CHANNELS):
                    df[f'label_{i}'] = 0
            
            # 3. 按照编码值排序，确保时序/空间连续性
            df = df.sort_values(by='encoding')
            channel_cols = [f'channel_{i}' for i in range(NUM_CHANNELS)]
            label_cols = [f'label_{i}' for i in range(NUM_CHANNELS)]
            
            # 检查列是否存在
            if not all(col in df.columns for col in channel_cols):
                print(f"文件 {csv_file.name} 缺少必要的通道列")
                continue

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
                
                # 获取中心行的 9 个通道标签 (index 127)
                center_labels = label_data[start + CENTER_IDX]
                
                self.samples.append(window_x)
                self.labels.append(center_labels)
                
                if not self.is_train:
                    self.file_info.append({
                        'file': str(csv_file),
                        'center_encoding': encodings[start + CENTER_IDX],
                        'start': start
                    })
        
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
        # 位置超向量 (256个，用于绑定)
        self.position_hvs = embeddings.Random(window_size, out_features)

    def forward(self, x):
        """
        前向传播编码逻辑：
        1. 数值编码 (Level Encoding): 将 0-255 的通道数值映射为高维超向量
        2. 通道绑定 (Channel Binding): 将数值超向量与对应的通道 ID 超向量进行绑定（异或操作）
        3. 通道聚合 (Channel Bundling): 在每个时间步，将 9 个通道的绑定结果聚合（叠加并归一化）
        4. 位置绑定 (Position Binding): 将每个时间步的聚合结果与其在窗口中的位置超向量绑定
        5. 窗口聚合 (Window Bundling): 将整个窗口 256 个位置的结果聚合，得到代表该窗口的超向量
        """
        # x: [batch, window_size, channels]
        batch_size = x.size(0)
        
        # 1. 数值编码
        # val_hvs: [batch, window_size, channels, out_features]
        val_hvs = self.value_hvs(x)
        
        # 2. 绑定通道信息 (使用 Bind 操作将数值与通道身份结合)
        # chan_hvs: [1, 1, channels, out_features]
        chan_hvs = self.channel_hvs.weight.unsqueeze(0).unsqueeze(0)
        # bound_hvs: [batch, window_size, channels, out_features]
        bound_hvs = torchhd.bind(val_hvs, chan_hvs)
        
        # 3. 聚合通道信息 (在每个采样点，合并 9 个通道的信息)
        # time_step_hvs: [batch, window_size, out_features]
        time_step_hvs = torchhd.multiset(bound_hvs)
        
        # 4. 绑定位置信息 (使用 Bind 操作将时间步信息与其在 256 窗口中的位置结合)
        # pos_hvs: [1, window_size, out_features]
        pos_hvs = self.position_hvs.weight.unsqueeze(0)
        # final_bound: [batch, window_size, out_features]
        final_bound = torchhd.bind(time_step_hvs, pos_hvs)
        
        # 5. 聚合整个窗口 (叠加所有位置的信息，得到窗口特征)
        # window_hv: [batch, out_features]
        window_hv = torchhd.multiset(final_bound)
        
        return torchhd.normalize(window_hv)

class HDCPointLevelModel:
    """
    HDC 点级分类模型管理类 (9 模型方案)
    每个通道拥有独立的分类器 (Centroid)
    """
    def __init__(self, dimensions=DIMENSIONS, device='cpu'):
        self.dimensions = dimensions
        self.device = device
        self.encoder = None
        self.models = nn.ModuleList([Centroid(dimensions, NUM_TYPES) for _ in range(NUM_CHANNELS)]).to(device)

    def train(self, train_loader):
        print("开始训练点级 HDC 模型 (9通道独立模型)...")
        # 初始化编码器
        try:
            first_batch = next(iter(train_loader))
            sample_shape = first_batch[0].shape[1:] # [window_size, channels]
            self.encoder = HDCPointLevelEncoder(self.dimensions, sample_shape[0]).to(self.device)
        except StopIteration:
            print("训练集为空，无法训练")
            return

        with torch.no_grad():
            for samples, targets in tqdm(train_loader, desc="训练中"):
                samples = samples.to(self.device)
                targets = targets.to(self.device) # [batch, 9]
                
                hv = self.encoder(samples) # [batch, dim]
                
                # 为每个通道的模型添加样本
                for ch in range(NUM_CHANNELS):
                    ch_targets = targets[:, ch]
                    self.models[ch].add(hv, ch_targets)
        
        # 归一化所有模型
        for ch in range(NUM_CHANNELS):
            self.models[ch].normalize()
        print("训练完成。")

    def predict(self, test_loader):
        print("开始预测 (9通道并行预测)...")
        if self.encoder is None:
            print("错误: 编码器未初始化")
            return []
            
        self.encoder.eval()
        for m in self.models:
            m.eval()
        
        all_preds = []
        with torch.no_grad():
            for samples, _ in tqdm(test_loader, desc="预测中"):
                samples = samples.to(self.device)
                hv = self.encoder(samples) # [batch, dim]
                
                batch_preds = []
                for ch in range(NUM_CHANNELS):
                    output = self.models[ch](hv, dot=True) # [batch, 20]
                    pred = torch.argmax(output, dim=1)
                    batch_preds.append(pred.unsqueeze(1))
                
                # 合并 9 个通道的预测结果 [batch, 9]
                combined_pred = torch.cat(batch_preds, dim=1)
                all_preds.append(combined_pred.cpu())
        
        return torch.cat(all_preds, dim=0).numpy()

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
            train_ld = data.DataLoader(train_ds, batch_size=32, shuffle=True)
            hdc_sys = HDCPointLevelModel(device=device)
            hdc_sys.train(train_ld)
            hdc_sys.save(model_path)
        else:
            print("训练集样本数为 0")
    
    # 2. 预测
    if os.path.exists(test_dir):
        test_ds = HDCPointLevelDataset(test_dir, is_train=False)
        if len(test_ds) > 0:
            test_ld = data.DataLoader(test_ds, batch_size=32, shuffle=False)
            
            hdc_sys = HDCPointLevelModel(device=device)
            if os.path.exists(model_path):
                hdc_sys.load(model_path)
                predictions = hdc_sys.predict(test_ld) # [N, 9]
                
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
                        'labels': pred_row # 9 个通道的预测值
                    })
                    
                for file_name, results in file_results.items():
                    # 读取原始数据以获取所有 encoding
                    df_orig = pd.read_csv(Path(test_dir) / file_name)
                    pred_df = pd.DataFrame({'encoding': df_orig['encoding']})
                    for i in range(NUM_CHANNELS):
                        pred_df[f'predicted_label_{i}'] = 0
                    
                    # 填充预测值
                    results_df = pd.DataFrame([
                        {'encoding': r['encoding'], **{f'predicted_label_{i}': r['labels'][i] for i in range(NUM_CHANNELS)}}
                        for r in results
                    ])
                    
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
