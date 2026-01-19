#!/usr/bin/env python3
"""
超声HDC算法测试脚本 - 改进版
使用分层抽样确保训练集和测试集中都有伤损点
对比两种数据输入方式：EMG风格（按段分割）vs 滑动窗口
"""

import sys
import os

# 添加torchhd库的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

import csv
from collections import Counter
import math
import statistics
import random

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

# ======================================
# 设备设置
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================================
# 配置参数
# ======================================
DIMENSIONS = 10000
NUM_LEVELS = 255
BATCH_SIZE = 1

# 测试参数组合
TEST_CONFIGS = [
    {'window': 32, 'step': 1, 'n_gram': 4},
    {'window': 32, 'step': 4, 'n_gram': 4},
    {'window': 32, 'step': 8, 'n_gram': 4},
    {'window': 64, 'step': 1, 'n_gram': 4},
    {'window': 64, 'step': 4, 'n_gram': 4},
    {'window': 64, 'step': 8, 'n_gram': 4},
    {'window': 128, 'step': 1, 'n_gram': 4},
    {'window': 128, 'step': 4, 'n_gram': 4},
    {'window': 128, 'step': 8, 'n_gram': 4},
    {'window': 256, 'step': 1, 'n_gram': 4},
    {'window': 256, 'step': 4, 'n_gram': 4},
    {'window': 256, 'step': 8, 'n_gram': 4},
]

# ======================================
# 数据加载
# ======================================
def load_data(csv_path):
    """加载CSV数据并应用简化标注规则"""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel_8 = float(row['channel_8'])
            # 简化标注规则：channel_8 > 170 → 伤损(label=1)，否则正常(label=0)
            label = 1 if channel_8 > 170 else 0
            data.append({
                'encoding': int(row['encoding']),
                'channel_8': channel_8,
                'label': label
            })
    return data

# ======================================
# 分层抽样划分数据
# ======================================
def stratified_split(data_list, test_ratio=0.3, random_seed=42):
    """分层抽样划分训练集和测试集
    
    确保训练集和测试集中都有伤损点和正常点
    """
    random.seed(random_seed)
    
    # 按标签分组
    label_0 = [item for item in data_list if item['label'] == 0]
    label_1 = [item for item in data_list if item['label'] == 1]
    
    print(f"正常点总数: {len(label_0)}")
    print(f"伤损点总数: {len(label_1)}")
    
    # 从每组中按比例抽取
    test_0 = random.sample(label_0, int(len(label_0) * test_ratio))
    test_1 = random.sample(label_1, int(len(label_1) * test_ratio))
    
    # 剩余的作为训练集
    train_0 = [item for item in label_0 if item not in test_0]
    train_1 = [item for item in label_1 if item not in test_1]
    
    # 合并
    train_data = train_0 + train_1
    test_data = test_0 + test_1
    
    # 打乱顺序
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print(f"\n分层抽样后:")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"  正常点: {len([d for d in train_data if d['label'] == 0])}")
    print(f"  伤损点: {len([d for d in train_data if d['label'] == 1])}")
    print(f"测试集: {len(test_data)} 个样本")
    print(f"  正常点: {len([d for d in test_data if d['label'] == 0])}")
    print(f"  伤损点: {len([d for d in test_data if d['label'] == 1])}")
    
    return train_data, test_data

# ======================================
# EMG风格数据集类（按段分割）
# ======================================
class EMGStyleDataset(data.Dataset):
    """EMG风格数据集：按标签变化点分割数据
    
    每个连续的标签段作为一个样本，如果段长度大于窗口大小，
    则按窗口大小切分，确保每个窗口内标签一致。
    """
    
    def __init__(self, data_list, window=256):
        self.window = window
        self.samples = []
        self.labels = []
        self._load_data(data_list)
    
    def _load_data(self, data_list):
        """按段分割数据"""
        channel_8 = [item['channel_8'] for item in data_list]
        labels = [item['label'] for item in data_list]
        
        # 找到标签变化点
        indexes = [i for i in range(1, len(labels)) if labels[i] != labels[i-1]]
        
        prev = 0
        for j in indexes:
            span = j - prev
            # 如果段长度大于窗口大小，按窗口切分
            if span > self.window:
                for k in range(math.floor(span / self.window)):
                    # 取窗口第一个点的标签
                    label = labels[prev + (self.window * k)]
                    # 取窗口数据
                    segment_data = channel_8[prev + (self.window * k): prev + (self.window * (k + 1))]
                    # 转换为 [window, 1] 形状
                    sample = torch.tensor(segment_data, dtype=torch.float32).reshape(-1, 1)
                    self.samples.append(sample)
                    self.labels.append(label)
            prev = j
        
        # 处理最后一段
        span = len(labels) - prev
        if span > self.window:
            for k in range(math.floor(span / self.window)):
                label = labels[prev + (self.window * k)]
                segment_data = channel_8[prev + (self.window * k): prev + (self.window * (k + 1))]
                sample = torch.tensor(segment_data, dtype=torch.float32).reshape(-1, 1)
                self.samples.append(sample)
                self.labels.append(label)
        
        print(f"EMG风格数据集: 生成 {len(self.samples)} 个样本")
        print(f"  标签分布: {Counter(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# ======================================
# 滑动窗口数据集类
# ======================================
class SlidingWindowDataset(data.Dataset):
    """滑动窗口数据集：固定窗口大小，滑动步长
    
    窗口标签 = 中心点标签
    """
    
    def __init__(self, data_list, window=32, step=1):
        self.window = window
        self.step = step
        self.samples = []
        self.labels = []
        self._load_data(data_list)
    
    def _load_data(self, data_list):
        """滑动窗口切分数据"""
        channel_8 = [item['channel_8'] for item in data_list]
        labels = [item['label'] for item in data_list]
        
        for start in range(0, len(channel_8) - self.window + 1, self.step):
            end = start + self.window
            window_data = channel_8[start:end]
            # 窗口标签 = 中心点标签
            center = (start + end) // 2
            label = labels[center]
            
            sample = torch.tensor(window_data, dtype=torch.float32).reshape(-1, 1)
            self.samples.append(sample)
            self.labels.append(label)
        
        print(f"滑动窗口数据集 (window={self.window}, step={self.step}): 生成 {len(self.samples)} 个样本")
        print(f"  标签分布: {Counter(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# ======================================
# 编码器类
# ======================================
class SimpleEncoder(nn.Module):
    """简单的单通道超维编码器"""
    
    def __init__(self, out_features, n_gram_size=4):
        super(SimpleEncoder, self).__init__()
        
        self.channels = embeddings.Random(1, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=200)
        self.n_gram_size = n_gram_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        signal = self.signals(input)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        samples = torchhd.multiset(samples)
        sample_hv = torchhd.ngrams(samples, n=self.n_gram_size)
        return torchhd.hard_quantize(sample_hv)

# ======================================
# 训练和测试函数
# ======================================
def train_and_test(train_ds, test_ds, n_gram_size=4):
    """训练和测试HDC模型"""
    
    # 初始化编码器
    sample_shape = train_ds[0][0].shape
    encode = SimpleEncoder(DIMENSIONS, n_gram_size=n_gram_size)
    encode = encode.to(device)
    
    # 初始化模型
    num_classes = 2  # 0: 正常, 1: 伤损
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    
    # 训练
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="训练中", leave=False):
            samples = samples.to(device)
            targets = targets.to(device)
            
            sample_hv = encode(samples)
            model.add(sample_hv, targets.long())
    
    # 归一化
    model.normalize()
    
    # 计算训练准确率
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="训练准确率", leave=False):
            samples = samples.to(device)
            targets = targets.to(device)
            
            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            pred_label = torch.argmax(output, dim=1)
            
            train_correct += (pred_label == targets).sum().item()
            train_total += targets.size(0)
    
    train_accuracy = (train_correct / train_total) * 100 if train_total > 0 else 0
    
    # 计算测试准确率
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_correct = 0
    test_total = 0
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    tn = 0  # True Negative
    
    with torch.no_grad():
        for samples, targets in tqdm(test_ld, desc="测试中", leave=False):
            samples = samples.to(device)
            targets = targets.to(device)
            
            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            pred_label = torch.argmax(output, dim=1)
            
            test_correct += (pred_label == targets).sum().item()
            test_total += targets.size(0)
            
            # 计算TP, FP, FN, TN
            for p, t in zip(pred_label, targets):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 1:
                    fn += 1
                else:
                    tn += 1
    
    test_accuracy = (test_correct / test_total) * 100 if test_total > 0 else 0
    
    # 计算精确率、召回率、F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

# ======================================
# 主测试函数
# ======================================
def main():
    print("=" * 80)
    print("超声HDC算法测试 - 改进版（使用分层抽样）")
    print("=" * 80)
    
    # 加载数据
    print("\n正在加载数据...")
    data_list = load_data(r"d:\HDC\data\train\接头3.cap_hdc.csv")
    print(f"总数据点数: {len(data_list)}")
    
    # 使用分层抽样划分数据
    print("\n正在划分训练集和测试集（分层抽样）...")
    train_data, test_data = stratified_split(data_list, test_ratio=0.3)
    
    # 测试结果汇总
    results = []
    
    # 测试EMG风格数据集
    print("\n" + "=" * 80)
    print("测试1: EMG风格数据集（按段分割）")
    print("=" * 80)
    
    for config in TEST_CONFIGS:
        print(f"\n配置: window={config['window']}, step={config['step']}, n_gram={config['n_gram']}")
        
        train_ds = EMGStyleDataset(train_data, window=config['window'])
        test_ds = EMGStyleDataset(test_data, window=config['window'])
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  跳过：数据集为空")
            continue
        
        metrics = train_and_test(train_ds, test_ds, n_gram_size=config['n_gram'])
        
        print(f"  训练准确率: {metrics['train_accuracy']:.2f}%")
        print(f"  测试准确率: {metrics['test_accuracy']:.2f}%")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        results.append({
            'method': 'EMG风格',
            'window': config['window'],
            'step': config['step'],
            'n_gram': config['n_gram'],
            **metrics
        })
    
    # 测试滑动窗口数据集
    print("\n" + "=" * 80)
    print("测试2: 滑动窗口数据集")
    print("=" * 80)
    
    for config in TEST_CONFIGS:
        print(f"\n配置: window={config['window']}, step={config['step']}, n_gram={config['n_gram']}")
        
        train_ds = SlidingWindowDataset(train_data, window=config['window'], step=config['step'])
        test_ds = SlidingWindowDataset(test_data, window=config['window'], step=config['step'])
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  跳过：数据集为空")
            continue
        
        metrics = train_and_test(train_ds, test_ds, n_gram_size=config['n_gram'])
        
        print(f"  训练准确率: {metrics['train_accuracy']:.2f}%")
        print(f"  测试准确率: {metrics['test_accuracy']:.2f}%")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        results.append({
            'method': '滑动窗口',
            'window': config['window'],
            'step': config['step'],
            'n_gram': config['n_gram'],
            **metrics
        })
    
    # 保存结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    # 找出最佳配置
    if results:
        best_result = max(results, key=lambda x: x['f1'])
        print(f"\n最佳配置（按F1分数排序）:")
        print(f"  方法: {best_result['method']}")
        print(f"  窗口大小: {best_result['window']}")
        print(f"  滑动步长: {best_result['step']}")
        print(f"  N-gram大小: {best_result['n_gram']}")
        print(f"  训练准确率: {best_result['train_accuracy']:.2f}%")
        print(f"  测试准确率: {best_result['test_accuracy']:.2f}%")
        print(f"  精确率: {best_result['precision']:.4f}")
        print(f"  召回率: {best_result['recall']:.4f}")
        print(f"  F1分数: {best_result['f1']:.4f}")
    
    # 按F1分数排序显示所有结果
    print("\n所有结果（按F1分数排序）:")
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    for i, r in enumerate(results_sorted[:10]):
        print(f"{i+1}. {r['method']}, window={r['window']}, step={r['step']}, n_gram={r['n_gram']}")
        print(f"   训练: {r['train_accuracy']:.2f}%, 测试: {r['test_accuracy']:.2f}%")
        print(f"   精确率: {r['precision']:.4f}, 召回率: {r['recall']:.4f}, F1: {r['f1']:.4f}")
        print(f"   TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, TN={r['tn']}")
    
    # 保存结果到CSV
    with open(r"d:\HDC\test_results_stratified.csv", 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['method', 'window', 'step', 'n_gram', 'train_accuracy', 
                     'test_accuracy', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n测试结果已保存到: d:\\HDC\\test_results_stratified.csv")

if __name__ == "__main__":
    main()
