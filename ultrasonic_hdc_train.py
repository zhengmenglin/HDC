#!/usr/bin/env python3
"""
超声波HDC训练脚本
训练超声波探伤HDC模型
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import argparse

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ultrasonic_hd_classifier import UltrasonicHDCClassifier


def load_csv_data(csv_file: str, window_size: int = 64, channels: int = 9):
    """
    从CSV文件加载数据

    Args:
        csv_file: CSV文件路径
        window_size: 窗口大小
        channels: 通道数

    Returns:
        samples: 样本列表
        labels: 标签列表
    """
    print(f"加载数据文件: {csv_file}")

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 计算特征数量 (总特征数 - 1个标签列)
    num_features = df.shape[1] - 1
    expected_features = channels * window_size * 3  # 9通道 × 64窗口 × 3特征

    if num_features != expected_features:
        print(f"警告: 特征数量不匹配，期望 {expected_features}，实际 {num_features}")

    samples = []
    labels = []

    for idx, row in df.iterrows():
        # 提取特征数据 (去掉标签列)
        features = row.values[:-1]

        # 重塑为 [channels, window_size, 3] 格式
        try:
            sample = features.reshape(channels, window_size, 3)
            samples.append(sample)
            labels.append(int(row.values[-1]))  # 标签
        except ValueError as e:
            print(f"警告: 样本 {idx} 重塑失败: {e}")
            continue

    print(f"加载完成: {len(samples)} 个样本")
    print(f"标签分布: {np.bincount(labels) if labels else '无标签'}")

    return samples, labels


def train_model(csv_files: list, model_output: str = None, d: int = 10000):
    """
    训练HDC模型

    Args:
        csv_files: CSV文件路径列表
        model_output: 模型输出路径
        d: 超向量维度
    """
    print("Starting training of ultrasonic HDC model...")

    # 1. 加载所有数据
    all_samples = []
    all_labels = []

    for csv_file in csv_files:
        samples, labels = load_csv_data(csv_file)
        all_samples.extend(samples)
        all_labels.extend(labels)

    if not all_samples:
        print("错误: 没有找到有效的训练数据")
        return None

    print(f"Total training data: {len(all_samples)} samples")

    # 2. 创建分类器
    print(f"Creating HDC classifier (dimension: {d})")
    classifier = UltrasonicHDCClassifier(d=d)

    # 3. 训练模型
    print("Starting training...")
    classifier.train(all_samples, all_labels)

    # 4. 保存模型
    if model_output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output = f"ultrasonic_hdc_model_{timestamp}.pth"

    classifier.save_model(model_output)

    print(f"Model training completed, saved to: {model_output}")
    return model_output


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='超声波HDC训练工具')
    parser.add_argument('--csv_files', nargs='+', help='训练CSV文件路径列表')
    parser.add_argument('--data_dir', help='包含CSV文件的目录')
    parser.add_argument('--model_output', help='模型输出路径')
    parser.add_argument('--d', type=int, default=10000, help='超向量维度')

    args = parser.parse_args()

    # 确定CSV文件列表
    csv_files = []
    if args.csv_files:
        csv_files = args.csv_files
    elif args.data_dir:
        data_path = Path(args.data_dir)
        csv_files = list(data_path.glob("*.csv"))
        csv_files = [str(f) for f in csv_files]
    else:
        print("错误: 请指定 --csv_files 或 --data_dir 参数")
        return

    if not csv_files:
        print("错误: 没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")

    # 训练模型
    model_path = train_model(csv_files, args.model_output, args.d)

    if model_path:
        print("\n训练成功完成！")
        print(f"模型文件: {model_path}")
    else:
        print("\n训练失败！")


if __name__ == "__main__":
    main()