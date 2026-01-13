#!/usr/bin/env python3
"""
测试完整的超声波HDC管道
包括数据转换、训练、预测和结果转换
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import shutil

# 添加项目路径
sys.path.append('.')

from ultrasonic_data_converter import UltrasonicDataConverter
from ultrasonic_hd_classifier import UltrasonicHDCClassifier
from ultrasonic_hdc_train import load_csv_data, train_model
from ultrasonic_hdc_predict import predict_with_model, convert_predictions_to_json

def test_full_pipeline():
    """测试完整管道"""
    print("=== 开始完整超声波HDC管道测试 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # 配置参数
    WINDOW_SIZE = 64
    STEP = 32
    DIMENSIONS = 10000
    TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
    TEST_DATA_DIR = Path(r"d:\HDC\data\test")
    OUTPUT_DIR = Path("pipeline_test_results")
    TEMP_DIR = Path("pipeline_temp")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    try:
        # 1. 数据转换：训练数据
        print("\n=== Step 1: 训练数据转换 ===")
        converter = UltrasonicDataConverter()
        
        # 获取所有训练数据文件
        train_files = []
        for ext in ['*.json', '*.xxc.json']:
            train_files.extend(TRAIN_DATA_DIR.glob(ext))
        
        # 转换训练数据
        train_csv_files = []
        for data_file in train_files:
            # 跳过标注文件
            if '标注' in data_file.name:
                continue
                
            # 查找对应的标注文件
            annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
            if not annotation_file.exists():
                print(f"警告: 未找到{data_file.name}的标注文件，跳过")
                continue
            
            # 检查是否已存在CSV文件
            output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            if output_csv.exists():
                print(f"✓ CSV文件已存在，跳过转换: {data_file.name}")
                train_csv_files.append(str(output_csv))
                continue
            
            # 转换文件
            try:
                result_file = converter.convert_single_file(
                    str(data_file), str(annotation_file), str(output_csv)
                )
                train_csv_files.append(str(output_csv))
                print(f"✓ 转换完成: {data_file.name}")
            except Exception as e:
                print(f"✗ 转换失败 {data_file.name}: {e}")
                continue
        
        if not train_csv_files:
            print("[FAIL] 没有成功转换的训练数据文件")
            return False
        
        print(f"\n训练数据转换结果: {len(train_csv_files)} 个文件成功转换或已存在")
        
        # 2. 数据转换：测试数据
        print("\n=== Step 2: 测试数据转换 ===")
        
        # 获取所有测试数据文件
        test_files = []
        for ext in ['*.json', '*.xxc.json']:
            test_files.extend(TEST_DATA_DIR.glob(ext))
        
        # 转换测试数据
        test_csv_files = []
        for data_file in test_files:
            # 跳过标注文件
            if '标注' in data_file.name:
                continue
                
            # 查找对应的标注文件
            annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
            if not annotation_file.exists():
                print(f"警告: 未找到{data_file.name}的标注文件，跳过")
                continue
            
            # 检查是否已存在CSV文件
            output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            if output_csv.exists():
                print(f"✓ CSV文件已存在，跳过转换: {data_file.name}")
                test_csv_files.append(str(output_csv))
                continue
            
            # 转换文件
            try:
                result_file = converter.convert_single_file(
                    str(data_file), str(annotation_file), str(output_csv)
                )
                test_csv_files.append(str(output_csv))
                print(f"✓ 转换完成: {data_file.name}")
            except Exception as e:
                print(f"✗ 转换失败 {data_file.name}: {e}")
                continue
        
        if not test_csv_files:
            print("[FAIL] 没有成功转换的测试数据文件")
            return False
        
        print(f"\n测试数据转换结果: {len(test_csv_files)} 个文件成功转换或已存在")
        
        # 3. 模型训练
        print("\n=== Step 3: 模型训练 ===")
        print(f"使用 {len(train_csv_files)} 个训练文件")
        print(f"超向量维度: {DIMENSIONS}")
        
        # 生成模型文件名
        model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = OUTPUT_DIR / f"hdc_model_{model_timestamp}.pth"
        
        try:
            trained_model_path = train_model(
                train_csv_files,
                model_output=str(model_file),
                d=DIMENSIONS
            )
            
            if not trained_model_path or not os.path.exists(trained_model_path):
                print("[FAIL] 模型训练失败")
                return False
            
            print(f"✓ 模型训练成功，保存路径: {trained_model_path}")
        except Exception as e:
            print(f"[FAIL] 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 模型预测
        print("\n=== Step 4: 模型预测 ===")
        print(f"使用 {len(test_csv_files)} 个测试文件")
        
        try:
            predictions = predict_with_model(
                str(model_file),
                test_csv_files,
                output_dir=str(OUTPUT_DIR)
            )
            
            if not predictions:
                print("[FAIL] 模型预测失败")
                return False
            
            print(f"✓ 模型预测完成，处理了 {len(predictions)} 个样本")
        except Exception as e:
            print(f"[FAIL] 模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. 结果评估
        print("\n=== Step 5: 结果评估 ===")
        
        # 统计预测结果
        pred_labels = [p['predicted_label'] for p in predictions]
        true_labels = [p['original_label'] for p in predictions]
        
        # 计算准确率
        correct = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)
        accuracy = correct / len(predictions) if predictions else 0
        
        # 统计各类别结果
        unique_labels = sorted(set(true_labels + pred_labels))
        label_stats = {}
        for label in unique_labels:
            total = true_labels.count(label)
            correct_pred = sum(1 for pred, true in zip(pred_labels, true_labels) if true == label and pred == true)
            precision = correct_pred / pred_labels.count(label) if pred_labels.count(label) > 0 else 0
            recall = correct_pred / total if total > 0 else 0
            
            label_stats[label] = {
                'total': total,
                'correct': correct_pred,
                'precision': precision,
                'recall': recall
            }
        
        # 输出评估结果
        print(f"测试样本总数: {len(predictions)}")
        print(f"准确率: {accuracy:.4f} ({correct}/{len(predictions)})")
        print("\n各类别详细统计:")
        print(f"{'标签':<8} {'总数':<6} {'正确':<6} {'精确率':<10} {'召回率':<10}")
        print("-"*40)
        for label, stats in label_stats.items():
            print(f"{label:<8} {stats['total']:<6} {stats['correct']:<6} {stats['precision']:<10.4f} {stats['recall']:<10.4f}")
        
        # 6. 结果转换为JSON
        print("\n=== Step 6: 结果转换为JSON ===")
        try:
            convert_predictions_to_json(
                predictions,
                str(TEST_DATA_DIR),
                str(OUTPUT_DIR / "json_results")
            )
            print(f"✓ JSON结果转换完成")
        except Exception as e:
            print(f"[WARN] JSON结果转换失败: {e}")
        

        
        # 输出最终结果
        print("\n" + "="*50)
        print("=== 完整超声波HDC管道测试完成 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型文件: {model_file}")
        print(f"预测结果目录: {OUTPUT_DIR}")
        print(f"准确率: {accuracy:.4f}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 确保临时目录被清理
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("\n✓ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)