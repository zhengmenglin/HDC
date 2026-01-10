#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试HDC算法代码修改
验证新增的角度矫正、通道管理等功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.HDC import (
    HyperdimensionalEncoder,
    BScanDataProcessor,
    calculate_actual_depth,
    calculate_encoding_offset,
    ChannelManager,
    ChannelConfig
)

def test_angle_correction():
    """测试角度矫正功能"""
    print("测试角度矫正功能...")
    
    # 测试数据：声程125mm，水平角度20°，俯仰角度70°
    acoustic_range = 125
    horizontal_angle = 20
    pitch_angle = 70
    
    # 计算实际深度
    actual_depth = calculate_actual_depth(acoustic_range, horizontal_angle, pitch_angle)
    print(f"声程: {acoustic_range}mm, 水平角度: {horizontal_angle}°, 俯仰角度: {pitch_angle}°")
    print(f"计算实际深度: {actual_depth:.2f}mm")
    
    # 计算编码偏差
    encoding_offset = calculate_encoding_offset(acoustic_range, horizontal_angle, pitch_angle)
    print(f"计算编码偏差: {encoding_offset:.2f}mm")
    
    assert actual_depth > 0, "实际深度计算错误"
    assert encoding_offset >= 0, "编码偏差计算错误"
    print("角度矫正功能测试通过！")

def test_channel_manager():
    """测试通道管理器"""
    print("\n测试通道管理器...")
    
    # 1. 测试默认初始化
    channel_manager = ChannelManager()
    channels = channel_manager.get_all_channels()
    print(f"默认初始化通道数: {len(channels)}")
    assert len(channels) == 9, "默认通道数错误"
    
    # 2. 测试获取通道配置
    channel_1 = channel_manager.get_channel_config(1)
    print(f"通道1配置: 俯仰角度={channel_1.pitch_angle}°, 水平角度={channel_1.horizontal_angle}°, 拼孔值={channel_1.offset}mm")
    assert channel_1 is not None, "无法获取通道1配置"
    
    # 3. 测试从配置数据初始化
    test_config = {
        "BeamAngle1": "75,15",
        "BeamOffset1": "15",
        "BeamAngle9": "5",
        "BeamOffset9": "75"
    }
    
    channel_manager = ChannelManager(test_config)
    updated_channel_1 = channel_manager.get_channel_config(1)
    updated_channel_9 = channel_manager.get_channel_config(9)
    
    print(f"更新后通道1配置: 俯仰角度={updated_channel_1.pitch_angle}°, 水平角度={updated_channel_1.horizontal_angle}°, 拼孔值={updated_channel_1.offset}mm")
    print(f"更新后通道9配置: 俯仰角度={updated_channel_9.pitch_angle}°, 水平角度={updated_channel_9.horizontal_angle}°, 拼孔值={updated_channel_9.offset}mm")
    
    assert updated_channel_1.pitch_angle == 75, "通道1俯仰角度更新错误"
    assert updated_channel_1.horizontal_angle == 15, "通道1水平角度更新错误"
    assert updated_channel_1.offset == 15, "通道1拼孔值更新错误"
    assert updated_channel_9.pitch_angle == 5, "通道9俯仰角度更新错误"
    assert updated_channel_9.offset == 75, "通道9拼孔值更新错误"
    
    print("通道管理器测试通过！")

def test_bscan_data_processor():
    """测试B显数据处理器"""
    print("\n测试B显数据处理器...")
    
    # 初始化编码器
    encoder = HyperdimensionalEncoder(d=1000, seed=42)
    
    # 生成测试配置数据
    test_config = {
        "E": 1,
        "BeamAngle1": "70,20",
        "BeamOffset1": "10"
    }
    
    # 初始化处理器
    processor = BScanDataProcessor(encoder, config_data=test_config)
    
    print("B显数据处理器初始化成功")
    print("B显数据处理器测试通过！")

def test_continuous_mapping():
    """测试连续映射功能"""
    print("\n测试连续映射功能...")
    
    encoder = HyperdimensionalEncoder(d=1000, seed=42)
    
    # 生成深度映射（0-300mm，301个级别）
    depth_mapping = encoder.continuous_mapping(0, 300, 301, 'depths')
    print(f"深度映射生成级别数: {len(depth_mapping)}")
    
    # 生成幅度映射（0-100%，101个级别）
    amplitude_mapping = encoder.continuous_mapping(0, 100, 101, 'amplitudes')
    print(f"幅度映射生成级别数: {len(amplitude_mapping)}")
    
    assert len(depth_mapping) == 301, "深度映射级别数错误"
    assert len(amplitude_mapping) == 101, "幅度映射级别数错误"
    print("连续映射功能测试通过！")

def main():
    """主测试函数"""
    print("开始测试HDC算法代码修改...")
    
    try:
        test_angle_correction()
        test_channel_manager()
        test_bscan_data_processor()
        test_continuous_mapping()
        
        print("\n所有测试通过！HDC算法代码修改验证成功！")
        return 0
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n测试出现异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
