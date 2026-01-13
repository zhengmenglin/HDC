#!/usr/bin/env python3
"""
超声波HDC识别脚本
使用训练好的模型对超声波数据进行伤损检测
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from ultrasonic_hd_classifier import UltrasonicHDCClassifier
from ultrasonic_data_converter import UltrasonicDataConverter


def load_csv_data(csv_file: str, window_size: int = 64, channels: int = 9):
    """
    从CSV文件加载数据

    Args:
        csv_file: CSV文件路径
        window_size: 窗口大小
        channels: 通道数

    Returns:
        samples: 样本列表
        metadata: 元数据（用于结果映射）
    """
    print(f"加载数据文件: {csv_file}")

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 计算特征数量 (总特征数 - 1个标签列)
    num_features = df.shape[1] - 1
    expected_features = channels * window_size * 3  # 9通道 × 64窗口 × 3特征

    samples = []
    metadata = []  # 存储每个样本的元信息

    for idx, row in df.iterrows():
        # 提取特征数据 (去掉标签列)
        features = row.values[:-1]

        # 重塑为 [channels, window_size, 3] 格式
        try:
            sample = features.reshape(channels, window_size, 3)
            samples.append(sample)

            # 存储元信息：样本索引、原始标签等
            meta = {
                'sample_idx': idx,
                'original_label': int(row.values[-1]) if not pd.isna(row.values[-1]) else 0
            }
            metadata.append(meta)

        except ValueError as e:
            print(f"警告: 样本 {idx} 重塑失败: {e}")
            continue

    print(f"加载完成: {len(samples)} 个样本")
    return samples, metadata


def predict_with_model(model_path: str, csv_files: List[str], output_dir: str = "results"):
    """
    使用训练好的模型进行预测

    Args:
        model_path: 模型文件路径
        csv_files: CSV文件路径列表
        output_dir: 输出目录

    Returns:
        results: 预测结果
    """
    print(f"加载模型: {model_path}")

    # 1. 加载模型
    classifier = UltrasonicHDCClassifier()
    classifier.load_model(model_path)

    # 2. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_results = []

    # 3. 处理每个CSV文件
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file}")

        # 加载数据
        samples, metadata = load_csv_data(csv_file)

        if not samples:
            print(f"警告: 文件 {csv_file} 中没有有效数据")
            continue

        # 批量预测
        predictions = classifier.predict_batch(samples)

        # 处理预测结果
        file_results = []
        for i, (pred_label, similarities) in enumerate(predictions):
            result = {
                'file': csv_file,
                'sample_idx': metadata[i]['sample_idx'],
                'predicted_label': pred_label,
                'predicted_name': classifier.get_class_name(pred_label),
                'original_label': metadata[i]['original_label'],
                'similarities': similarities
            }
            file_results.append(result)

        # 保存单个文件的预测结果
        csv_name = Path(csv_file).stem
        result_file = output_path / f"{csv_name}_predictions.csv"

        # 创建结果DataFrame
        result_data = []
        for result in file_results:
            row = {
                'sample_idx': result['sample_idx'],
                'predicted_label': result['predicted_label'],
                'predicted_name': result['predicted_name'],
                'original_label': result['original_label'],
                'max_similarity': max(result['similarities'].values())
            }
            # 添加每个类别的相似度
            for label, sim in result['similarities'].items():
                row[f'sim_{label}'] = sim

            result_data.append(row)

        result_df = pd.DataFrame(result_data)
        result_df.to_csv(result_file, index=False)

        print(f"文件 {csv_name} 预测完成: {len(file_results)} 个样本")
        print(f"预测结果保存到: {result_file}")

        all_results.extend(file_results)

    return all_results


def convert_predictions_to_json(predictions: List[Dict], original_data_dir: str,
                               output_dir: str = "json_results"):
    """
    将预测结果转换为JSON标注格式

    Args:
        predictions: 预测结果列表
        original_data_dir: 原始数据目录
        output_dir: 输出目录
    """
    print("转换预测结果为JSON格式...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 按文件分组预测结果
    file_groups = {}
    for pred in predictions:
        file_path = pred['file']
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append(pred)

    converter = UltrasonicDataConverter()

    # 处理每个文件
    for csv_file, file_predictions in file_groups.items():
        csv_path = Path(csv_file)
        original_file = None

        # 找到对应的原始数据文件
        for ext in ['.json', '.xxc']:
            candidate = csv_path.parent / csv_path.name.replace('_hdc.csv', ext)
            if candidate.exists():
                original_file = candidate
                break

        if not original_file:
            print(f"警告: 找不到文件 {csv_path} 对应的原始数据文件")
            continue

        print(f"处理文件: {original_file}")

        # 加载原始数据
        data_list, config = converter.load_json_data(str(original_file))

        # 创建新的标注结果
        annotations = []

        # 处理预测结果，将伤损预测转换为标注格式
        damage_predictions = [p for p in file_predictions if p['predicted_label'] > 0]

        for pred in damage_predictions:
            sample_idx = pred['sample_idx']

            # 计算对应的编码位置范围 (反推窗口位置)
            window_start = sample_idx * 32  # step=32
            window_end = window_start + 64   # window_size=64

            # 创建标注
            annotation = {
                "id": f"damage_{sample_idx}",
                "type": "伤损",
                "subtype": {
                    "id": pred['predicted_label'],
                    "name": pred['predicted_name']
                },
                "points": [],
                "confidence": pred['similarities'].get(pred['predicted_label'], 0)
            }

            # 为该窗口范围内的编码位置创建点
            for encoding in range(window_start, min(window_end, 256)):
                # 找到该编码位置的数据
                for item in data_list:
                    if item.get('E') == encoding + 1 and 'B1' in item:
                        b1_data = item['B1']
                        if len(b1_data) >= 3:
                            point = {
                                "e": encoding + 1,
                                "channel": int(b1_data[0]),
                                "depth": float(b1_data[1]),
                                "amplitude": float(b1_data[2])
                            }
                            annotation["points"].append(point)
                        break

            if annotation["points"]:  # 只添加有数据点的标注
                annotations.append(annotation)

        # 创建完整的标注文件
        annotation_data = {
            "annotations": annotations,
            "metadata": {
                "source_file": str(original_file),
                "prediction_model": "UltrasonicHDC",
                "total_predictions": len(file_predictions),
                "damage_predictions": len(annotations)
            }
        }

        # 保存JSON文件
        json_filename = f"{original_file.stem}_hdc_result.json"
        json_file = output_path / json_filename

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)

        print(f"JSON结果保存到: {json_file}")
        print(f"检测到 {len(annotations)} 个伤损区域")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='超声波HDC识别工具')
    parser.add_argument('--model', required=True, help='模型文件路径')
    parser.add_argument('--csv_files', nargs='+', help='预测CSV文件路径列表')
    parser.add_argument('--data_dir', help='包含CSV文件的目录')
    parser.add_argument('--output_dir', default='results', help='输出目录')
    parser.add_argument('--convert_json', action='store_true',
                       help='是否转换结果为JSON格式')
    parser.add_argument('--original_data_dir', help='原始数据目录（用于JSON转换）')

    args = parser.parse_args()

    # 确定CSV文件列表
    csv_files = []
    if args.csv_files:
        csv_files = args.csv_files
    elif args.data_dir:
        data_path = Path(args.data_dir)
        csv_files = list(data_path.glob("*_hdc.csv"))
        csv_files = [str(f) for f in csv_files]
    else:
        print("错误: 请指定 --csv_files 或 --data_dir 参数")
        return

    if not csv_files:
        print("错误: 没有找到CSV文件")
        return

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")

    # 进行预测
    predictions = predict_with_model(args.model, csv_files, args.output_dir)

    if predictions:
        print(f"\n预测完成！共处理 {len(predictions)} 个样本")

        # 统计预测结果
        pred_labels = [p['predicted_label'] for p in predictions]
        unique_labels, counts = np.unique(pred_labels, return_counts=True)

        print("预测结果统计:")
        for label, count in zip(unique_labels, counts):
            label_name = predictions[0]['predicted_name'] if predictions[0]['predicted_label'] == label else f"标签_{label}"
            print(f"  {label_name} ({label}): {count} 个")

        # 转换为JSON格式（如果指定）
        if args.convert_json and args.original_data_dir:
            convert_predictions_to_json(predictions, args.original_data_dir,
                                      args.output_dir)
    else:
        print("\n预测失败！")


if __name__ == "__main__":
    main()