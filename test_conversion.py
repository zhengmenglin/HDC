#!/usr/bin/env python3
"""
测试超声波数据转换功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from ultrasonic_data_converter import UltrasonicDataConverter

def test_conversion():
    """测试数据转换功能"""
    print("Testing ultrasonic data conversion...")

    # 创建转换器
    converter = UltrasonicDataConverter()

    # 测试文件路径
    data_file = r"d:\HDC\data\train\170921_16229片段.xxc.json"
    annotation_file = r"d:\HDC\data\train\170921_16229片段.xxc标注.json"
    output_file = "test_conversion.csv"

    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return

    try:
        # 执行转换
        result_file = converter.convert_single_file(
            data_file, annotation_file, output_file,
            window_size=64, step=32
        )

        print(f"Conversion successful! Output: {result_file}")

        # 检查输出文件
        if os.path.exists(result_file):
            file_size = os.path.getsize(result_file)
            print(f"Output file size: {file_size} bytes")
        else:
            print("Output file was not created")

    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversion()