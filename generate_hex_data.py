#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成示例十六进制数据文件和标签文件
"""

import numpy as np
import random

def float_to_hex(f):
    """
    将浮点数转换为十六进制字符串
    """
    return ''.join(f'{b:02x}' for b in np.float32(f).tobytes())

def generate_train_data():
    """
    生成训练数据文件和标签文件
    """
    # 训练数据文件
    train_data_file = open('train_data.hex', 'w')
    # 标签文件
    train_labels_file = open('train_labels.txt', 'w')
    
    # 伤损类型和正常结构
    flaw_types = ["轨头外侧核伤", "轨头中心核伤", "轨头内侧核伤", "接头", "螺孔", "导线孔"]
    
    # 为每种类型生成5个样本
    for i, flaw_type in enumerate(flaw_types):
        for j in range(5):
            # 生成随机数据
            channel = i % 9  # 0-8通道
            encoding = j * 10  # 0, 10, 20, 30, 40
            
            if "核伤" in flaw_type:
                # 伤损样本：dB值较高
                depth = random.uniform(30, 60)
                db_value = random.uniform(40, 60)
            else:
                # 正常结构：dB值较低
                depth = random.uniform(0, 20)
                db_value = random.uniform(10, 30)
            
            # 转换为十六进制
            channel_hex = f'{channel:02x}'
            encoding_hex = f'{encoding:04x}'
            depth_hex = float_to_hex(depth)
            db_hex = float_to_hex(db_value)
            
            # 写入数据文件
            train_data_file.write(f'{channel_hex} {encoding_hex} {depth_hex} {db_hex}\n')
            
            # 写入标签文件
            train_labels_file.write(f'{channel_hex} {encoding_hex} {flaw_type}\n')
    
    # 关闭文件
    train_data_file.close()
    train_labels_file.close()
    print("训练数据文件和标签文件生成完成")

def generate_test_data():
    """
    生成测试数据文件
    """
    # 测试数据文件
    test_data_file = open('test_data.hex', 'w')
    
    # 生成10个测试样本
    for i in range(10):
        # 生成随机数据
        channel = i % 9  # 0-8通道
        encoding = i * 10 + 5  # 5, 15, 25, 35, 45, 55, 65, 75, 85, 95
        
        # 随机生成伤损或正常样本
        if random.random() < 0.5:
            # 伤损样本：dB值较高
            depth = random.uniform(30, 60)
            db_value = random.uniform(40, 60)
        else:
            # 正常结构：dB值较低
            depth = random.uniform(0, 20)
            db_value = random.uniform(10, 30)
        
        # 转换为十六进制
        channel_hex = f'{channel:02x}'
        encoding_hex = f'{encoding:04x}'
        depth_hex = float_to_hex(depth)
        db_hex = float_to_hex(db_value)
        
        # 写入数据文件
        test_data_file.write(f'{channel_hex} {encoding_hex} {depth_hex} {db_hex}\n')
    
    # 关闭文件
    test_data_file.close()
    print("测试数据文件生成完成")

if __name__ == "__main__":
    generate_train_data()
    generate_test_data()
    print("所有示例数据文件生成完成")
