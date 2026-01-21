#!/usr/bin/env python3
"""
测试标注文件转换
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultrasonic_data_converter import UltrasonicDataConverter

def test_annotation_conversion():
    """测试标注文件转换"""
    converter = UltrasonicDataConverter()
    
    input_file = "d:\\HDC\\data\\test\\接p.json"
    output_file = "d:\\HDC\\data\\test\\接p_with_labels.csv"
    
    print(f"开始转换文件（带标注）: {input_file}")
    
    try:
        # 使用convert_single_file函数，不提供标注文件
        result_file = converter.convert_single_file(input_file, None, output_file)
        print(f"转换成功！输出文件: {result_file}")
        
        # 读取并显示结果摘要
        import pandas as pd
        df = pd.read_csv(result_file)
        print(f"\n转换结果摘要:")
        print(f"输出文件形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"编码范围: {df['encoding'].min()} 到 {df['encoding'].max()}")
        print(f"非零数据点数量: {(df.iloc[:, 1:10] != 0).sum().sum()}")
        
        # 显示前10行数据
        print(f"\n前10行数据:")
        print(df.head(10))
        
        # 显示各通道的数据统计
        print(f"\n各通道数据统计:")
        for i in range(9):
            channel_col = f'channel_{i}'
            label_col = f'label_{i}'
            non_zero_count = (df[channel_col] != 0).sum()
            label_count = (df[label_col] != 0).sum()
            print(f"通道 {i}: {non_zero_count} 个非零数据点, {label_count} 个标注点")
        
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_annotation_conversion()
    if success:
        print("\n测试通过！")
    else:
        print("\n测试失败！")