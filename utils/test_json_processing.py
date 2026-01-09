#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试JSON数据处理功能
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_reader import HexDataReader
from hdc_ultrasonic_bscan import BScanDataProcessor

def test_json_data_processing():
    """测试JSON数据处理功能"""
    print("测试JSON数据处理功能...")
    
    # 创建测试数据读取器
    reader = HexDataReader()
    
    # 测试数据文件路径
    test_file = "d:\\HDC\\test_b显.json"
    
    # 读取数据文件
    print(f"读取测试文件: {test_file}")
    data_points = reader.read_data_file(test_file)
    
    if not data_points:
        print("错误：未能读取到数据")
        return False
    
    print(f"成功读取到 {len(data_points)} 个数据点")
    
    # 打印前5个数据点
    print("\n前5个数据点：")
    for i, point in enumerate(data_points[:5]):
        print(f"数据点 {i+1}: 通道={point.channel}, 编码={point.encoding}, 深度={point.depth}mm, dB值={point.db_value:.2f}dB, 幅度={point.amplitude:.1f}%")
    
    # 创建B显数据处理器
    bscan_processor = BScanDataProcessor()
    
    # 初始化B显矩阵
    max_encoding = max(point.encoding for point in data_points) + 1
    print(f"\n初始化B显矩阵，最大编码值：{max_encoding}")
    bscan_processor.init_bscan_matrix(max_encoding=max_encoding)
    
    # 更新B显矩阵
    print("\n更新B显矩阵...")
    for data_point in data_points:
        bscan_processor.update_bscan_matrix(
            data_point.channel, 
            data_point.encoding, 
            data_point.depth, 
            data_point.db_value,
            data_point.amplitude
        )
    
    # 获取B显数据（深度值、dB值、幅度值）
    depth_data = bscan_processor.get_bscan_data(data_type="depth")
    db_data = bscan_processor.get_bscan_data(data_type="db")
    amplitude_data = bscan_processor.get_bscan_data(data_type="amplitude")
    
    print(f"\nB显数据矩阵形状：通道数={len(depth_data)}, 编码数={len(depth_data[0])}")
    
    # 验证幅度值数据
    print("\n验证幅度值数据：")
    has_amplitude_data = False
    for channel in range(3):  # 仅检查前3个通道
        for encoding in range(min(5, len(depth_data[channel]))):  # 检查前5个编码值
            amp_value = amplitude_data[channel][encoding]
            if amp_value > 0:
                has_amplitude_data = True
                print(f"通道{channel+1}, 编码{encoding}: 幅度值={amp_value:.1f}%")
    
    if not has_amplitude_data:
        print("警告：未检测到有效的幅度值数据")
    
    print("\nJSON数据处理测试完成！")
    return True

if __name__ == "__main__":
    test_json_data_processing()
