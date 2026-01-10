#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试文件查找功能
"""
import os

def test_find_json_files():
    """测试文件查找功能"""
    json_files = []
    
    # 定义要搜索的文件夹，按优先级排序（test优先于train）
    folders = ['test', 'train']
    
    # 构造data目录路径
    data_dir = os.path.join(os.getcwd(), 'data')
    print(f"正在data目录下查找: {data_dir}")
    
    for folder in folders:
        # 构造完整的文件夹路径
        folder_path = os.path.join(data_dir, folder)
        print(f"正在查找文件夹: {folder_path}")
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"找到文件夹: {folder_path}")
            # 遍历文件夹中的所有文件
            try:
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    # 检查文件是否为.json文件且不是标注文件
                    if filename.endswith('.json') and '标注' not in filename:
                        json_files.append(file_path)
                        print(f"找到文件: {file_path}")
            except Exception as e:
                print(f"遍历文件夹 {folder_path} 时出错: {str(e)}")
        else:
            print(f"文件夹不存在或不是目录: {folder_path}")
    
    # 对文件列表进行排序
    json_files.sort()
    print(f"最终文件列表: {json_files}")
    
    return json_files

if __name__ == "__main__":
    files = test_find_json_files()
    if files:
        print(f"共找到 {len(files)} 个文件")
    else:
        print("没有找到任何文件")
