#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试标注保存功能
"""

import os
import tempfile

# 测试自动生成标注文件名
def test_annotation_filename_generation():
    """测试标注文件名的自动生成"""

    # 模拟原始文件名
    test_cases = [
        "170921_16229片段.xxc.json",
        "data.json",
        "test_data.json"
    ]

    for original_filename in test_cases:
        print(f"\n测试文件名: {original_filename}")

        # 模拟文件路径
        json_dir = tempfile.gettempdir()
        json_file_path = os.path.join(json_dir, original_filename)

        # 生成标注文件名
        name_without_ext = os.path.splitext(original_filename)[0]
        annotation_filename = f"{name_without_ext}标注.json"

        print(f"标注文件名: {annotation_filename}")

        # 如果文件已存在，添加数字后缀
        counter = 1
        while os.path.exists(os.path.join(json_dir, annotation_filename)):
            name_without_ext_counter = f"{name_without_ext}_{counter}"
            annotation_filename = f"{name_without_ext_counter}标注.json"
            counter += 1

        annotation_file_path = os.path.join(json_dir, annotation_filename)
        print(f"完整路径: {annotation_file_path}")

if __name__ == "__main__":
    test_annotation_filename_generation()