#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载是否正确获取编码值和通道信息
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.HDC import BScanDataProcessor, HyperdimensionalEncoder

# 初始化编码器
encoder = HyperdimensionalEncoder(d=10000, seed=42)
# 使用连续映射生成深度和幅度的数值向量
encoder.continuous_mapping(min_val=0, max_val=300, num_levels=301, base_key='depths')
encoder.continuous_mapping(min_val=0, max_val=100, num_levels=101, base_key='amplitudes')

# 初始化数据处理器
processor = BScanDataProcessor(encoder)

# 加载测试数据
test_file = 'd:\\HDC\\data\\test\\200922_17387片段.xxc.json'
print(f"加载测试文件: {test_file}")

samples, labels, config_data = processor.load_json_data(test_file)
print(f"加载到 {len(samples)} 个B显数据点")

# 分析样本数据
if samples:
    # 统计通道分布
    channel_counts = {}
    for sample in samples:
        channel, code, depth, amplitude, db_value = sample
        if channel not in channel_counts:
            channel_counts[channel] = 0
        channel_counts[channel] += 1
    
    print(f"通道分布: {channel_counts}")
    
    # 统计编码值分布
    code_counts = {}
    for sample in samples:
        channel, code, depth, amplitude, db_value = sample
        if code not in code_counts:
            code_counts[code] = 0
        code_counts[code] += 1
    
    print(f"编码值分布: {len(code_counts)} 个不同编码值")
    if len(code_counts) > 0:
        min_code = min(code_counts.keys())
        max_code = max(code_counts.keys())
        print(f"编码值范围: {min_code} - {max_code}")
    
    # 打印前10个样本的详细信息
    print("\n前10个样本详细信息:")
    for i, sample in enumerate(samples[:10]):
        channel, code, depth, amplitude, db_value = sample
        print(f"样本 {i+1}: 通道={channel}, 编码={code}, 深度={depth:.2f}, 幅度={amplitude:.2f}")
else:
    print("没有加载到样本数据")
