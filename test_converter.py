#!/usr/bin/env python3
"""
测试超声波数据转换器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultrasonic_data_converter import UltrasonicDataConverter

def test_converter():
    """测试转换器功能"""
    converter = UltrasonicDataConverter()
    
    # 创建测试JSON数据
    test_json_data = [
        {
            "E": 1,
            "BeamOffset": [0, 35, 1, 229, 2, 115, 3, 175, 4, 123, 5, 150, 6, 20, 7, 268, 8, 279],
            "RailType": "60",
            "RailHgt": 176
        },
        {
            "E": 100,
            "B1": [0.0, 62.0, 100.0, 16.61, 14.57]
        },
        {
            "E": 101,
            "B1": [1.0, 65.0, 100.0, 18.61, 15.57]
        },
        {
            "E": 102,
            "B1": [2.0, 68.0, 100.0, 20.61, 16.57]
        },
        {
            "E": 200,
            "B1": [3.0, 70.0, 100.0, 22.61, 17.57]
        },
        {
            "E": 201,
            "B1": [4.0, 72.0, 100.0, 24.61, 18.57]
        },
        {
            "E": 202,
            "B1": [5.0, 75.0, 100.0, 26.61, 19.57]
        },
        {
            "E": 300,
            "B1": [6.0, 78.0, 100.0, 28.61, 20.57]
        },
        {
            "E": 301,
            "B1": [7.0, 80.0, 100.0, 30.61, 21.57]
        },
        {
            "E": 302,
            "B1": [8.0, 82.0, 100.0, 32.61, 22.57]
        }
    ]
    
    # 保存测试JSON文件
    import json
    test_file = "test_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_json_data, f, ensure_ascii=False, indent=2)
    
    print("测试数据已创建")
    
    # 测试转换函数
    try:
        output_file = converter.convert_to_csv_with_beam_offset(test_file)
        print(f"转换成功，输出文件: {output_file}")
        
        # 读取并检查输出文件
        import pandas as pd
        df = pd.read_csv(output_file)
        print(f"输出文件形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行数据:")
        print(df.head())
        
        # 检查数据完整性
        print(f"编码范围: {df['encoding'].min()} 到 {df['encoding'].max()}")
        print(f"非零数据点数量: {(df.iloc[:, 1:] != 0).sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    success = test_converter()
    if success:
        print("测试通过！")
    else:
        print("测试失败！")