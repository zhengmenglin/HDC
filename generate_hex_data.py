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
    flaw_types = ["轨头外侧核伤", "轨头中心核伤", "轨头内侧核伤", "轨头水平分裂", 
                 "轨腰水平裂", "螺孔裂纹", "接头", "螺孔", "导线孔", "焊缝"]
    
    # 为每种类型生成连续的伤损区域
    for i, flaw_type in enumerate(flaw_types):
        # 生成5个连续伤损区域
        for region in range(3):
            # 随机选择起始通道和编码值
            channel = i % 9  # 0-8通道
            start_encoding = region * 20  # 0, 20, 40
            
            # 生成连续的3-5个点，形成一个伤损区域
            num_points = random.randint(3, 5)
            base_depth = random.uniform(30, 60) if "核伤" in flaw_type or "裂" in flaw_type else random.uniform(0, 20)
            
            for j in range(num_points):
                encoding = start_encoding + j
                
                # 伤损点在深度上连续变化
                depth_variation = random.uniform(-5, 5)  # 深度变化范围
                depth = base_depth + depth_variation
                
                if "核伤" in flaw_type or "裂" in flaw_type:
                    # 伤损样本：dB值较高
                    db_value = random.uniform(40, 60)
                else:
                    # 正常结构：dB值较低
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
    
    # 生成一些无法识别的点（dB值介于伤损和正常之间）
    for i in range(20):
        channel = random.randint(0, 8)  # 0-8通道
        encoding = random.randint(0, 99)  # 0-99编码值
        depth = random.uniform(15, 45)  # 深度介于正常和伤损之间
        db_value = random.uniform(25, 40)  # dB值介于正常和伤损之间
        
        # 转换为十六进制
        channel_hex = f'{channel:02x}'
        encoding_hex = f'{encoding:04x}'
        depth_hex = float_to_hex(depth)
        db_hex = float_to_hex(db_value)
        
        # 写入数据文件
        train_data_file.write(f'{channel_hex} {encoding_hex} {depth_hex} {db_hex}\n')
        
        # 写入标签文件，标记为"其他"
        train_labels_file.write(f'{channel_hex} {encoding_hex} 其他\n')
    
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
    
    # 伤损类型和正常结构
    flaw_types = ["轨头外侧核伤", "轨头中心核伤", "轨头内侧核伤", "轨头水平分裂", 
                 "轨腰水平裂", "螺孔裂纹", "接头", "螺孔", "导线孔", "焊缝"]
    
    # 生成连续伤损区域
    for i, flaw_type in enumerate(flaw_types):
        # 每个类型生成2个连续伤损区域
        for region in range(2):
            # 随机选择起始通道和编码值
            channel = i % 9  # 0-8通道
            start_encoding = random.randint(0, 80)  # 0-80编码值
            
            # 生成连续的3-5个点，形成一个伤损区域
            num_points = random.randint(3, 5)
            base_depth = random.uniform(30, 60) if "核伤" in flaw_type or "裂" in flaw_type else random.uniform(0, 20)
            
            for j in range(num_points):
                encoding = start_encoding + j
                
                # 伤损点在深度上连续变化
                depth_variation = random.uniform(-5, 5)  # 深度变化范围
                depth = base_depth + depth_variation
                
                if "核伤" in flaw_type or "裂" in flaw_type:
                    # 伤损样本：dB值较高
                    db_value = random.uniform(40, 60)
                else:
                    # 正常结构：dB值较低
                    db_value = random.uniform(10, 30)
                
                # 转换为十六进制
                channel_hex = f'{channel:02x}'
                encoding_hex = f'{encoding:04x}'
                depth_hex = float_to_hex(depth)
                db_hex = float_to_hex(db_value)
                
                # 写入数据文件
                test_data_file.write(f'{channel_hex} {encoding_hex} {depth_hex} {db_hex}\n')
    
    # 生成一些无法识别的点（dB值介于伤损和正常之间）
    for i in range(15):
        channel = random.randint(0, 8)  # 0-8通道
        encoding = random.randint(0, 99)  # 0-99编码值
        depth = random.uniform(15, 45)  # 深度介于正常和伤损之间
        db_value = random.uniform(25, 40)  # dB值介于正常和伤损之间
        
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
