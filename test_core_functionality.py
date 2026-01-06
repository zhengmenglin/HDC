#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试核心功能：JSON数据读取、B显矩阵更新、分类器训练和测试
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_reader import HexDataReader
from hdc_ultrasonic_bscan import BScanDataProcessor, RailFlawClassifier

def test_core_functionality():
    """测试核心功能"""
    print("测试核心功能...")
    
    # 创建数据读取器
    reader = HexDataReader()
    
    # 测试数据文件
    train_file = "d:\\HDC\\train_b显.json"
    test_file = "d:\\HDC\\test_b显.json"
    labels_file = "d:\\HDC\\train_labels.txt"
    
    # 1. 测试数据读取
    print(f"\n1. 测试数据读取...")
    train_data_points = reader.read_data_file(train_file)
    test_data_points = reader.read_data_file(test_file)
    
    print(f"   训练数据：{len(train_data_points)} 个数据点")
    print(f"   测试数据：{len(test_data_points)} 个数据点")
    
    # 2. 测试数据转换
    print(f"\n2. 测试数据转换...")
    training_data = reader.convert_to_training_data(train_data_points, labels_file)
    test_data = reader.convert_to_test_data(test_data_points)
    
    print(f"   训练数据：{len(training_data)} 类")
    print(f"   测试数据：{len(test_data)} 个样本")
    
    # 3. 测试B显矩阵处理
    print(f"\n3. 测试B显矩阵处理...")
    bscan_processor = BScanDataProcessor()
    
    # 初始化B显矩阵
    max_encoding = max(max(point.encoding for point in train_data_points), 
                      max(point.encoding for point in test_data_points)) + 1
    bscan_processor.init_bscan_matrix(max_encoding=max_encoding)
    
    # 更新B显矩阵
    for data_point in train_data_points + test_data_points:
        bscan_processor.update_bscan_matrix(
            data_point.channel, 
            data_point.encoding, 
            data_point.depth, 
            data_point.db_value,
            data_point.amplitude
        )
    
    # 获取B显数据
    depth_data = bscan_processor.get_bscan_data(data_type="depth")
    amplitude_data = bscan_processor.get_bscan_data(data_type="amplitude")
    
    print(f"   B显矩阵：{len(depth_data)} 通道, {len(depth_data[0])} 编码值")
    
    # 4. 测试分类器训练和测试
    print(f"\n4. 测试分类器训练和测试...")
    flaw_classifier = RailFlawClassifier()
    
    # 训练分类器
    flaw_classifier.train_classifier(training_data)
    print(f"   分类器训练完成")
    
    # 测试分类器
    correct_predictions = 0
    for i, sample in enumerate(test_data):
        result, max_sim, all_sims = flaw_classifier.classify_flaw(sample)
        print(f"   样本 {i+1}: 预测类别={result}, 相似度={max_sim:.4f}")
        correct_predictions += 1  # 简化测试，不计算准确率
    
    print(f"   测试完成：{len(test_data)} 个样本")
    
    # 5. 测试伤损聚类
    print(f"\n5. 测试伤损聚类...")
    clusters = bscan_processor.cluster_flaws(eps=5.0, min_samples=2)
    print(f"   发现 {len(clusters)} 个伤损聚类")
    
    # 6. 测试特征提取
    print(f"\n6. 测试特征提取...")
    bscan_data = bscan_processor.get_bscan_points()
    features = flaw_classifier.extract_features(bscan_data, 0, 0)
    print(f"   提取的特征：{list(features.keys())}")
    
    print(f"\n✅ 核心功能测试完成！")
    return True

if __name__ == "__main__":
    test_core_functionality()
