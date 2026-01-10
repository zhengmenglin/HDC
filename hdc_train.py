#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDC算法训练脚本
用于从训练数据和标注数据中训练HDC模型
"""

import os
import sys
import json
import pickle
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.HDC import HyperdimensionalEncoder, HDCClassifier, BScanDataProcessor

def train_model():
    """
    训练HDC模型
    """
    print("开始HDC模型训练...")
    
    # 1. 配置训练参数
    config = {
        'd': 10000,  # 减小超维向量维度，加快训练速度
        'seed': 42,  # 随机种子
        'min_depth': 0.0,  # 最小深度值
        'max_depth': 255.0,  # 最大深度值
        'min_amplitude': 0.0,  # 最小幅度值
        'max_amplitude': 200.0,  # 最大幅度值
        'num_depth_levels': 301,  # 减小深度离散级别数，加快训练速度
        'num_amplitude_levels': 51,  # 减小幅度离散级别数，加快训练速度
        'iterations': 5,  # 重训练迭代次数
        'learning_rate': 0.1  # 学习率
    }
    
    # 2. 准备训练数据和标注数据路径
    train_dir = os.path.join('data', 'train')
    
    # 初始化编码器和数据处理器
    print("初始化HDC编码器...")
    encoder = HyperdimensionalEncoder(d=config['d'], seed=config['seed'])
    
    print("初始化B显数据处理器...")
    processor = BScanDataProcessor(encoder)
    
    # 3. 查找train目录下所有的训练数据文件
    import glob
    all_json_files = glob.glob(os.path.join(train_dir, '*.json'))
    
    # 过滤掉标注文件，只保留原始数据文件
    train_files = []
    for file_path in all_json_files:
        file_name = os.path.basename(file_path)
        if "标注" not in file_name:
            train_files.append(file_path)
    
    if not train_files:
        print(f"错误：在 {train_dir} 目录中没有找到训练数据文件")
        return
    
    print(f"找到 {len(train_files)} 个训练数据文件")
    
    # 4. 加载所有训练数据和标注数据
    all_samples = []
    all_damage_regions = []
    all_normal_regions = []
    
    for train_file in train_files:
        print(f"\n处理训练文件：{train_file}")
        
        # 加载训练数据
        samples, _, config_data = processor.load_json_data(train_file)
        all_samples.extend(samples)
        print(f"  加载到 {len(samples)} 个B显数据点，累计 {len(all_samples)} 个")
        
        # 查找对应的标注文件
        file_name = os.path.basename(train_file)
        name_without_ext = os.path.splitext(file_name)[0]
        annotation_file = os.path.join(train_dir, f"{name_without_ext}标注.json")
        
        if os.path.exists(annotation_file):
            print(f"  加载标注数据：{annotation_file}")
            damage_regions, normal_regions = processor.load_annotation_data(annotation_file)
            all_damage_regions.extend(damage_regions)
            all_normal_regions.extend(normal_regions)
            print(f"  加载到 {len(damage_regions)} 个伤损区域和 {len(normal_regions)} 个正常区域")
        else:
            print(f"  警告：标注文件 {annotation_file} 不存在，跳过标注加载")
    
    # 5. 统计最终数据量
    print(f"\n数据加载完成，总计：")
    print(f"  训练数据点：{len(all_samples)}")
    print(f"  伤损区域：{len(all_damage_regions)}")
    print(f"  正常区域：{len(all_normal_regions)}")
    
    # 使用合并后的数据
    samples = all_samples
    damage_regions = all_damage_regions
    normal_regions = all_normal_regions
    
    # 如果标注数据为空，自动提取潜在伤损区域
    if not damage_regions and not normal_regions:
        print("标注数据为空，尝试自动提取潜在伤损区域...")
        damage_regions = []
        potential_regions = processor.extract_damage_regions_from_bscan(samples)
        for region in potential_regions:
            damage_regions.append((region, "潜在伤损"))
        
        # 添加一些正常区域（假设非伤损区域为正常）
        normal_regions = [([(1, 0, 50, 30)], "正常区域")]  # 示例正常区域，注意是点列表
    
    print(f"提取到 {len(damage_regions)} 个伤损区域和 {len(normal_regions)} 个正常区域")
    
    # 6. 使用连续映射生成深度和幅度的数值向量
    print("生成连续映射的数值向量...")
    encoder.continuous_mapping(config['min_depth'], config['max_depth'], 
                              config['num_depth_levels'], 'depths')
    encoder.continuous_mapping(config['min_amplitude'], config['max_amplitude'], 
                              config['num_amplitude_levels'], 'amplitudes')
    
    # 7. 初始化分类器
    print("初始化HDC分类器...")
    classifier = HDCClassifier(encoder)
    classifier.set_depth_range(config['min_depth'], config['max_depth'])
    classifier.set_amplitude_range(config['min_amplitude'], config['max_amplitude'])
    
    # 8. 使用区域进行训练
    print("开始使用区域数据训练模型...")
    classifier.train_with_regions(damage_regions, normal_regions)
    
    # 9. 转换样本格式用于重训练
    training_samples = [(s[0], s[1], s[2], s[3]) for s in samples]  # 只保留需要的字段
    
    # 自动生成标签（如果没有标注）
    if not damage_regions and not normal_regions:
        print("自动生成训练标签...")
        labels = []
        for sample in training_samples:
            channel, code, depth, amplitude = sample
            # 简单的标签生成逻辑
            if amplitude > 60:
                labels.append("潜在伤损")
            else:
                labels.append("正常区域")
        
        # 进行重训练优化
        print("开始重训练优化...")
        classifier.retrain(training_samples, labels, 
                          iterations=config['iterations'], 
                          learning_rate=config['learning_rate'])
    
    # 10. 保存训练好的模型
    model_dir = os.path.join('models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 生成模型文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"hdc_model_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    print(f"保存模型到：{model_path}")
    
    # 保存模型
    with open(model_path, 'wb') as f:
        pickle.dump({
            'encoder': encoder,
            'classifier': classifier,
            'config': config,
            'timestamp': timestamp
        }, f)
    
    # 保存配置到JSON文件
    config_filename = f"hdc_config_{timestamp}.json"
    config_path = os.path.join(model_dir, config_filename)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("HDC模型训练完成！")
    print(f"模型文件：{model_path}")
    print(f"配置文件：{config_path}")
    
    return model_path

if __name__ == "__main__":
    train_model()
