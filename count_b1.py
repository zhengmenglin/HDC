#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计JSON文件中B1字段数量的工具脚本

用法:
    python count_b1.py [directory]
    
参数:
    directory   - 要统计的目录路径，默认为当前目录
"""

import os
import json
import sys
import glob

def count_b1_in_json(data, count=0):
    """
    递归统计JSON数据中B1字段的数量
    
    Args:
        data: JSON数据（字典或列表）
        count: 当前统计的数量
        
    Returns:
        统计的B1字段数量
    """
    if isinstance(data, dict):
        # 如果是字典，检查是否包含B1字段
        if 'B1' in data:
            count += 1
        # 递归检查字典的值
        for value in data.values():
            count = count_b1_in_json(value, count)
    elif isinstance(data, list):
        # 如果是列表，递归检查列表的每个元素
        for item in data:
            count = count_b1_in_json(item, count)
    
    return count

def main():
    """
    主函数
    """
    # 获取命令行参数，默认当前目录
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '.'
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在")
        return 1
    
    # 获取目录下所有JSON文件
    json_files = glob.glob(os.path.join(directory, '*.json'))
    
    if not json_files:
        print(f"在目录 {directory} 中没有找到JSON文件")
        return 1
    
    print(f"在目录 {directory} 中找到 {len(json_files)} 个JSON文件")
    print("=" * 60)
    
    # 统计每个文件中的B1数量
    total_count = 0
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计B1数量
            b1_count = count_b1_in_json(data)
            total_count += b1_count
            
            # 输出结果
            file_name = os.path.basename(file_path)
            print(f"{file_name}: {b1_count} 个B1")
            
        except json.JSONDecodeError as e:
            print(f"错误：文件 {os.path.basename(file_path)} 不是有效的JSON文件")
            print(f"  错误信息: {e}")
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错")
            print(f"  错误信息: {e}")
    
    print("=" * 60)
    print(f"总计: {total_count} 个B1")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
