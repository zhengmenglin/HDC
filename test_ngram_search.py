#!/usr/bin/env python3
"""
N_GRAM_SIZE参数搜索测试
测试N_GRAM_SIZE从1到32的所有值，找出最佳配置
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

# 固定参数（基于之前测试的最佳配置）
FIXED_WINDOW = 256
FIXED_STEP = 8

# N_GRAM_SIZE搜索范围
N_GRAM_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

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
    """分层抽样划分训练集和测试集"""
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
# 滑动窗口数据集类
# ======================================
class SlidingWindowDataset(data.Dataset):
    """滑动窗口数据集：固定窗口大小，滑动步长"""
    
    def __init__(self, data_list, window=256, step=8):
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
    print("N_GRAM_SIZE参数搜索测试")
    print("=" * 80)
    print(f"固定参数: window={FIXED_WINDOW}, step={FIXED_STEP}")
    print(f"测试范围: N_GRAM_SIZE = {N_GRAM_RANGE[0]} 到 {N_GRAM_RANGE[-1]}")
    
    # 加载数据
    print("\n正在加载数据...")
    data_list = load_data(r"d:\HDC\data\train\接头3.cap_hdc.csv")
    print(f"总数据点数: {len(data_list)}")
    
    # 使用分层抽样划分数据
    print("\n正在划分训练集和测试集（分层抽样）...")
    train_data, test_data = stratified_split(data_list, test_ratio=0.3)
    
    # 测试结果汇总
    results = []
    
    # 循环测试不同的N_GRAM_SIZE
    print("\n" + "=" * 80)
    print("开始测试不同的N_GRAM_SIZE值")
    print("=" * 80)
    
    for n_gram in N_GRAM_RANGE:
        print(f"\n{'='*80}")
        print(f"测试 N_GRAM_SIZE = {n_gram}")
        print(f"{'='*80}")
        
        # 创建数据集
        train_ds = SlidingWindowDataset(train_data, window=FIXED_WINDOW, step=FIXED_STEP)
        test_ds = SlidingWindowDataset(test_data, window=FIXED_WINDOW, step=FIXED_STEP)
        
        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  跳过：数据集为空")
            continue
        
        # 训练和测试
        metrics = train_and_test(train_ds, test_ds, n_gram_size=n_gram)
        
        print(f"\n结果:")
        print(f"  训练准确率: {metrics['train_accuracy']:.2f}%")
        print(f"  测试准确率: {metrics['test_accuracy']:.2f}%")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        
        results.append({
            'n_gram': n_gram,
            **metrics
        })
    
    # 保存结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    # 按F1分数排序
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print(f"\n最佳N_GRAM_SIZE（按F1分数排序）:")
    best = results_sorted[0]
    print(f"  N_GRAM_SIZE: {best['n_gram']}")
    print(f"  训练准确率: {best['train_accuracy']:.2f}%")
    print(f"  测试准确率: {best['test_accuracy']:.2f}%")
    print(f"  精确率: {best['precision']:.4f}")
    print(f"  召回率: {best['recall']:.4f}")
    print(f"  F1分数: {best['f1']:.4f}")
    print(f"  TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}")
    
    # 显示前10名
    print(f"\n前10名（按F1分数排序）:")
    for i, r in enumerate(results_sorted[:10]):
        print(f"{i+1:2d}. N_GRAM={r['n_gram']:2d} | "
              f"训练={r['train_accuracy']:6.2f}% | "
              f"测试={r['test_accuracy']:6.2f}% | "
              f"精确率={r['precision']:.4f} | "
              f"召回率={r['recall']:.4f} | "
              f"F1={r['f1']:.4f}")
    
    # 显示所有结果
    print(f"\n所有结果（按N_GRAM_SIZE排序）:")
    results_by_ngram = sorted(results, key=lambda x: x['n_gram'])
    for r in results_by_ngram:
        print(f"N_GRAM={r['n_gram']:2d} | "
              f"训练={r['train_accuracy']:6.2f}% | "
              f"测试={r['test_accuracy']:6.2f}% | "
              f"精确率={r['precision']:.4f} | "
              f"召回率={r['recall']:.4f} | "
              f"F1={r['f1']:.4f}")
    
    # 保存结果到CSV
    with open(r"d:\HDC\test_ngram_results.csv", 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['n_gram', 'train_accuracy', 'test_accuracy', 
                     'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n测试结果已保存到: d:\\HDC\\test_ngram_results.csv")

if __name__ == "__main__":
    main()
