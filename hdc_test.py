#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDC算法测试脚本
用于使用训练好的模型处理测试数据并生成标注JSON文件
"""

import os
import sys
import json
import pickle
import glob
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.HDC import BScanDataProcessor

def find_latest_model(model_dir='models'):
    """
    查找最新的模型文件
    Args:
        model_dir: 模型目录
    Returns:
        最新模型文件路径
    """
    model_files = glob.glob(os.path.join(model_dir, 'hdc_model_*.pkl'))
    if not model_files:
        return None
    # 按修改时间排序，获取最新的模型文件
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

def load_model(model_path):
    """
    加载HDC模型
    Args:
        model_path: 模型文件路径
    Returns:
        encoder, classifier, config
    """
    print(f"加载模型：{model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    encoder = model_data['encoder']
    classifier = model_data['classifier']
    config = model_data['config']
    
    print(f"模型加载完成，超维向量维度：{config['d']}")
    return encoder, classifier, config

def process_test_file(test_file, encoder, classifier, config):
    """
    处理单个测试文件
    Args:
        test_file: 测试数据文件路径
        encoder: 超维编码器
        classifier: HDC分类器
        config: 模型配置
    Returns:
        检测到的伤损区域列表
    """
    print(f"处理测试文件：{test_file}")
    
    # 初始化数据处理器
    processor = BScanDataProcessor(encoder)
    
    # 加载测试数据，添加错误处理
    try:
        samples, _, config_data = processor.load_json_data(test_file)
    except json.JSONDecodeError as e:
        print(f"警告：测试文件 {test_file} 不是有效的JSON文件，跳过处理")
        print(f"错误信息：{e}")
        return []
    
    if not samples:
        print(f"警告：测试文件 {test_file} 中没有有效数据")
        return []
    
    print(f"加载到 {len(samples)} 个B显数据点")
    
    # 使用滑动窗口扫描检测伤损
    print("使用滑动窗口扫描检测伤损...")
    detected_regions = classifier.scan_with_sliding_window(
        samples, 
        window_size=15, 
        step=2, 
        min_similarity=0.52
    )
    
    print(f"检测到 {len(detected_regions)} 个潜在伤损区域")
    
    return detected_regions

def generate_annotation_json(detected_regions, test_file):
    """
    生成标注JSON文件
    Args:
        detected_regions: 检测到的伤损区域列表
        test_file: 测试数据文件路径
    Returns:
        标注JSON数据
    """
    annotation_data = []
    
    for i, (region_points, predicted_label, similarity) in enumerate(detected_regions):
        # 转换为标注格式
        annotation_item = {
            "id": i + 1,
            "type": "damage" if "伤损" in predicted_label else "normal",
            "label": predicted_label,
            "confidence": float(similarity),
            "points": [list(point) for point in region_points],
            # 添加边界框信息
            "bounding_box": {
                "min_channel": min(point[0] for point in region_points),
                "max_channel": max(point[0] for point in region_points),
                "min_code": min(point[1] for point in region_points),
                "max_code": max(point[1] for point in region_points),
                "min_depth": min(point[2] for point in region_points),
                "max_depth": max(point[2] for point in region_points)
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        annotation_data.append(annotation_item)
    
    return annotation_data

def save_annotation_file(annotation_data, test_file, output_dir='data/test'):
    """
    保存标注文件
    Args:
        annotation_data: 标注JSON数据
        test_file: 测试数据文件路径
        output_dir: 输出目录
    Returns:
        保存的文件路径
    """
    # 生成输出文件名：原始文件名 + "标注" + ".json"
    base_name = os.path.basename(test_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}标注.json"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"保存标注文件到：{output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)
    
    return output_path

def test_model(model_path=None):
    """
    测试HDC模型
    Args:
        model_path: 模型文件路径，默认为最新模型
    """
    print("开始HDC模型测试...")
    
    # 1. 查找或加载指定模型
    if not model_path:
        model_path = find_latest_model()
        if not model_path:
            print("错误：未找到模型文件，请先训练模型或指定模型路径")
            return
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在")
        return
    
    # 2. 加载模型
    encoder, classifier, config = load_model(model_path)
    
    # 3. 准备测试数据路径
    test_data_dir = os.path.join('data', 'test')
    
    # 如果测试目录不存在，创建它
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
        print(f"创建测试目录：{test_data_dir}")
        print("请将测试数据文件放入该目录后重新运行")
        return
    
    # 4. 获取所有测试数据文件
    # 获取所有.json文件，过滤掉标注文件
    all_json_files = glob.glob(os.path.join(test_data_dir, '*.json'))
    test_files = [f for f in all_json_files if "标注" not in os.path.basename(f)]
    
    if not test_files:
        print(f"警告：测试目录 {test_data_dir} 中没有找到测试数据文件")
        print("请将测试数据文件放入该目录")
        return
    
    print(f"找到 {len(test_files)} 个测试数据文件")
    
    # 5. 处理每个测试文件
    for test_file in test_files:
        # 处理测试文件
        detected_regions = process_test_file(test_file, encoder, classifier, config)
        
        # 生成标注JSON
        annotation_data = generate_annotation_json(detected_regions, test_file)
        
        # 保存标注文件
        save_annotation_file(annotation_data, test_file, test_data_dir)
    
    print("HDC模型测试完成！")
    print("请查看 data/test 目录下的标注文件")

if __name__ == "__main__":
    # 支持命令行参数指定模型路径
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    test_model(model_path)
