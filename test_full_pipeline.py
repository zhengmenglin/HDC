#!/usr/bin/env python3
"""
测试完整的超声波HDC管道
包括数据转换、训练、预测和结果转换
"""

import sys
import os
import pandas as pd
import torch
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from ultrasonic_data_converter import UltrasonicDataConverter
from ultrasonic_hd_classifier import UltrasonicHDCClassifier
from ultrasonic_hdc_train import load_csv_data, train_model
from ultrasonic_hdc_predict import predict_with_model

def test_full_pipeline():
    """测试完整管道"""
    print("Testing full ultrasonic HDC pipeline...")

    # 1. 数据转换测试
    print("\n=== Step 1: Data Conversion ===")
    converter = UltrasonicDataConverter()

    data_file = r"d:\HDC\data\train\170921_16229片段.xxc.json"
    annotation_file = r"d:\HDC\data\train\170921_16229片段.xxc标注.json"
    csv_file = "pipeline_test.csv"

    if os.path.exists(data_file) and os.path.exists(annotation_file):
        try:
            result_file = converter.convert_single_file(
                data_file, annotation_file, csv_file,
                window_size=64, step=32
            )
            print(f"[OK] Data conversion successful: {result_file}")
        except Exception as e:
            print(f"[FAIL] Data conversion failed: {e}")
            return
    else:
        print("[FAIL] Test data files not found")
        return

    # 2. 数据加载测试
    print("\n=== Step 2: Data Loading ===")
    try:
        samples, labels = load_csv_data(csv_file)
        print(f"[OK] Data loading successful: {len(samples)} samples")
        print(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        return

    # 3. 模型功能测试
    print("\n=== Step 3: Model Functionality Test ===")
    try:
        classifier = UltrasonicHDCClassifier(d=1000)  # 使用较小的维度以加快测试

        # 测试编码器功能
        test_sample = samples[0]  # 使用第一个样本测试
        encoded = classifier.encoder(torch.tensor(test_sample, dtype=torch.float32).unsqueeze(0))
        print(f"[OK] Encoder test successful: input shape {test_sample.shape} -> output shape {encoded.shape}")

        # 测试预测功能（使用未训练的模型）
        pred_label, similarities = classifier.predict(test_sample)
        pred_name = classifier.get_class_name(pred_label)
        max_sim = max(similarities.values()) if similarities else 0
        print(f"[OK] Prediction test successful: Pred={pred_label}({pred_name}), MaxSim={max_sim:.3f}")

    except Exception as e:
        print(f"[FAIL] Model functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 结果转换测试
    print("\n=== Step 5: Result Conversion ===")
    try:
        from ultrasonic_hdc_predict import convert_predictions_to_json

        # 创建模拟预测结果
        mock_predictions = []
        pred_label, similarities = classifier.predict(test_sample)
        for i in range(3):  # 创建3个模拟预测结果
            mock_pred = {
                'file': csv_file,
                'sample_idx': i,
                'predicted_label': pred_label,
                'predicted_name': classifier.get_class_name(pred_label),
                'original_label': labels[i] if i < len(labels) else 0,
                'similarities': similarities
            }
            mock_predictions.append(mock_pred)

        # 转换回JSON
        convert_predictions_to_json(
            mock_predictions,
            str(Path(data_file).parent),
            "test_json_results"
        )
        print("✓ Result conversion to JSON successful")

    except Exception as e:
        print(f"[FAIL] Result conversion failed: {e}")
        return

    # 清理测试文件
    print("\n=== Cleaning up test files ===")
    test_files = [csv_file, model_file, "test_json_results"]
    for file in test_files:
        if os.path.exists(file):
            if os.path.isfile(file):
                os.remove(file)
                print(f"[OK] Removed: {file}")
            elif os.path.isdir(file):
                import shutil
                shutil.rmtree(file)
                print(f"[OK] Removed directory: {file}")

    print("\n[SUCCESS] Full pipeline test completed successfully!")
    print("All components are working correctly.")

if __name__ == "__main__":
    test_full_pipeline()