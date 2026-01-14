#!/usr/bin/env python3
"""
检查标注文件是否存在，并验证命名格式
"""

import os
from pathlib import Path

def check_annotation_files(directory):
    """检查目录中的标注文件"""
    print(f"检查目录: {directory}")
    
    # 获取目录中的所有文件
    files = list(Path(directory).glob("*"))
    
    # 按文件名排序
    files.sort(key=lambda x: x.name)
    
    print("\n所有文件:")
    for file in files:
        print(f"  {file.name}")
    
    # 检查标注文件
    annotation_files = [f for f in files if "标注" in f.name]
    print(f"\n找到 {len(annotation_files)} 个标注文件:")
    for file in annotation_files:
        print(f"  {file.name}")
    
    # 检查数据文件和对应的标注文件
    print("\n数据文件和对应的标注文件:")
    data_files = [f for f in files if f.suffix in [".json", ".xxc.json"] and "标注" not in f.name]
    
    for data_file in data_files:
        print(f"\n数据文件: {data_file.name}")
        
        # 检查预期的标注文件名格式
        expected_annotation_name1 = f"{data_file.stem}标注{data_file.suffix}"
        expected_annotation_path1 = data_file.parent / expected_annotation_name1
        print(f"  预期标注文件名1: {expected_annotation_name1}")
        print(f"  存在: {expected_annotation_path1.exists()}")
        
        # 检查另一种可能的标注文件名格式
        expected_annotation_name2 = f"{data_file.stem.replace('.xxc', '')}标注{data_file.suffix}"
        expected_annotation_path2 = data_file.parent / expected_annotation_name2
        print(f"  预期标注文件名2: {expected_annotation_name2}")
        print(f"  存在: {expected_annotation_path2.exists()}")

if __name__ == "__main__":
    # 检查train目录
    check_annotation_files(r"d:\HDC\data\train")
    
    # 检查test目录
    check_annotation_files(r"d:\HDC\data\test")
