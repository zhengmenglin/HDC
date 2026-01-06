#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试JSON格式数据读取
"""

from data_reader import HexDataReader

def test_json_reader():
    """测试JSON格式数据读取"""
    print("测试JSON格式数据读取...")
    
    # 创建数据读取器
    reader = HexDataReader()
    
    # 读取JSON格式的B显数据
    data_points = reader.read_data_file('test_b显.json')
    
    # 打印读取到的数据点
    print(f"读取到 {len(data_points)} 个数据点")
    for i, data_point in enumerate(data_points):
        print(f"数据点{i+1}: 通道={data_point.channel}, 编码={data_point.encoding}, 深度={data_point.depth}, dB值={data_point.db_value}, 幅度={data_point.amplitude}%")
    
    # 测试转换为测试数据
    test_data = reader.convert_to_test_data(data_points)
    print(f"转换为 {len(test_data)} 个测试样本")
    for i, sample in enumerate(test_data):
        print(f"测试样本{i+1}: {sample}")

if __name__ == "__main__":
    test_json_reader()
