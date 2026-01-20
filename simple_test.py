"""
简化的测试脚本 - 不依赖torchhd
"""

import os
import json
import csv
import numpy as np
from data_adapter import DataAdapter
from config import config

def simple_test():
    """简化的测试 - 只测试数据转换"""
    print("="*60)
    print("超声波探伤HDC系统 - 简化测试")
    print("="*60)
    
    # 创建数据适配器
    adapter = DataAdapter()
    
    # 1. 转换训练数据
    print("\n步骤1: 转换训练数据")
    train_dir = config.get("data.train_dir", "data/train")
    train_files = []
    
    for filename in os.listdir(train_dir):
        if filename.endswith('.json') and not filename.endswith('标注.json'):
            json_file = os.path.join(train_dir, filename)
            base_name = os.path.splitext(filename)[0]
            annotation_file = os.path.join(train_dir, f'{base_name}标注.json')
            
            # 转换文件
            output_csv = adapter.convert_to_csv(
                adapter.load_original_json(json_file),
                annotation_file if os.path.exists(annotation_file) else None,
                os.path.join(config.get("data.output_dir", "output"), f'{base_name}_converted.csv')
            )
            train_files.append(output_csv)
    
    print(f"已转换 {len(train_files)} 个训练文件")
    
    # 2. 转换测试数据
    print("\n步骤2: 转换测试数据")
    test_dir = config.get("data.test_dir", "data/test")
    test_files = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.json') and not filename.endswith('标注.json'):
            json_file = os.path.join(test_dir, filename)
            base_name = os.path.splitext(filename)[0]
            annotation_file = os.path.join(test_dir, f'{base_name}标注.json')
            
            # 转换文件
            output_csv = adapter.convert_to_csv(
                adapter.load_original_json(json_file),
                annotation_file if os.path.exists(annotation_file) else None,
                os.path.join(config.get("data.output_dir", "output"), f'{base_name}_converted.csv')
            )
            test_files.append(output_csv)
    
    print(f"已转换 {len(test_files)} 个测试文件")
    
    # 3. 合并训练数据
    print("\n步骤3: 合并训练数据")
    output_dir = config.get("data.output_dir", "output")
    train_csv = os.path.join(output_dir, "train_merged.csv")
    adapter.merge_csv_files(train_files, train_csv)
    
    # 4. 合并测试数据
    print("\n步骤4: 合并测试数据")
    test_csv = os.path.join(output_dir, "test_merged.csv")
    adapter.merge_csv_files(test_files, test_csv)
    
    # 5. 加载并分析数据
    print("\n步骤5: 分析数据")
    train_data, train_labels = adapter.load_csv(train_csv)
    test_data, test_labels = adapter.load_csv(test_csv)
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试标签形状: {test_labels.shape}")
    
    # 统计标签分布
    print("\n训练标签分布:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  标签 {label}: {count} 个样本 ({count/len(train_labels)*100:.2f}%)")
    
    print("\n测试标签分布:")
    unique, counts = np.unique(test_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  标签 {label}: {count} 个样本 ({count/len(test_labels)*100:.2f}%)")
    
    # 6. 数据统计
    print("\n数据统计:")
    print(f"  训练样本数: {len(train_data)}")
    print(f"  测试样本数: {len(test_data)}")
    print(f"  总样本数: {len(train_data) + len(test_data)}")
    print(f"  特征数: {train_data.shape[1]}")
    print(f"  类别数: {len(np.unique(np.concatenate([train_labels, test_labels])))}")
    
    # 7. 检查数据质量
    print("\n数据质量检查:")
    
    # 检查缺失值
    train_nan = np.isnan(train_data).sum()
    test_nan = np.isnan(test_data).sum()
    print(f"  训练数据缺失值: {train_nan}")
    print(f"  测试数据缺失值: {test_nan}")
    
    # 检查深度值范围
    depth_cols = train_data[:, 1:10]  # 9个通道的深度值
    print(f"  深度值范围: [{np.min(depth_cols):.2f}, {np.max(depth_cols):.2f}]")
    print(f"  深度值均值: {np.mean(depth_cols):.2f}")
    print(f"  深度值标准差: {np.std(depth_cols):.2f}")
    
    print("\n" + "="*60)
    print("简化测试完成！")
    print("="*60)
    
    return {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels
    }

if __name__ == "__main__":
    simple_test()