#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查JSON文件中每个"E"字段所在的{}对象是否包含"B1"字段
如果某个"E"所在的对象没有"B1"字段，记录该"E"的值并保存结果

用法:
    python check_e_without_b1.py [input_directory] [output_file]
    
参数:
    input_directory   - 要检查的JSON文件目录，默认为当前目录
    output_file       - 结果输出文件，默认为"e_without_b1.txt"
"""

import os
import json
import sys
import glob

def check_e_without_b1(data, file_name):
    """
    检查JSON数据中每个"E"字段所在的对象是否包含"B1"字段
    
    Args:
        data: JSON数据
        file_name: 文件名
        
    Returns:
        符合条件的"E"值列表，每个元素为(dict_index, e_value)
    """
    result = []
    
    if isinstance(data, list):
        # 遍历列表中的每个对象
        for index, item in enumerate(data):
            if isinstance(item, dict):
                # 检查是否包含"E"字段
                if 'E' in item:
                    # 检查是否不包含"B1"字段
                    if 'B1' not in item:
                        e_value = item['E']
                        result.append((index, e_value))
    
    return result

def main():
    """
    主函数
    """
    # 获取命令行参数
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
    else:
        input_directory = '.'
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'e_without_b1.txt'
    
    # 检查目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误：目录 {input_directory} 不存在")
        return 1
    
    # 获取目录下所有JSON文件
    json_files = glob.glob(os.path.join(input_directory, '*.json'))
    
    if not json_files:
        print(f"在目录 {input_directory} 中没有找到JSON文件")
        return 1
    
    print(f"在目录 {input_directory} 中找到 {len(json_files)} 个JSON文件")
    print("=" * 60)
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 遍历每个JSON文件
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查文件
                e_list = check_e_without_b1(data, os.path.basename(file_path))
                
                # 输出结果
                file_name = os.path.basename(file_path)
                if e_list:
                    print(f"文件 {file_name} 中找到 {len(e_list)} 个没有B1的E")
                    # 写入输出文件
                    out_f.write(f"===== {file_name} =====\n")
                    for index, e_value in e_list:
                        out_f.write(f"索引 {index}: E = {e_value}\n")
                    out_f.write("\n")
                else:
                    print(f"文件 {file_name} 中没有找到没有B1的E")
                    
            except json.JSONDecodeError as e:
                print(f"错误：文件 {os.path.basename(file_path)} 不是有效的JSON文件")
                print(f"  错误信息: {e}")
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时出错")
                print(f"  错误信息: {e}")
    
    print("=" * 60)
    print(f"检查完成，结果已保存到 {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
