#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点级 HDC 分类器完整测试流程脚本
包含：
1. 数据清理与初始化
2. JSON 到 CSV 数据转换 (利用 UltrasonicDataConverter)
3. HDC 模型预测
4. 性能指标计算 (Precision, Recall, F1-score)
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加 torchhd 库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

# 导入 HDC 模型相关类
from hdc_point_level_classifier import (
    HDCPointLevelDataset, 
    HDCPointLevelModel, 
    WINDOW_SIZE, 
    NUM_CHANNELS, 
    NUM_TYPES
)
# 导入数据转换器
from ultrasonic_data_converter import UltrasonicDataConverter

# ======================================
# 配置参数
# ======================================
TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
TEST_DATA_DIR = Path(r"d:\HDC\data\test")
MODEL_PATH = Path(r"d:\HDC\models\hdc_point_level_9ch.pth")
RESULTS_DIR = Path(r"d:\HDC\hdc_test_results")

# ======================================
# 工具函数
# ======================================
def clean_old_files(data_dirs):
    """清理旧的中间文件，保持环境整洁"""
    print("\n[INFO] 正在清理旧的 CSV 和预测文件...")
    patterns = ["*_hdc.csv", "*_label.csv", "*_encoding.csv", "*_predicted_label.csv"]
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for pattern in patterns:
            for file_path in data_dir.glob(pattern):
                try:
                    file_path.unlink()
                except Exception:
                    pass

def get_json_files(data_dir):
    """获取目录下的原始 JSON 数据文件（排除标注文件）"""
    if not data_dir.exists():
        return []
    data_files = []
    for ext in ['*.json', '*.xxc.json']:
        data_files.extend(data_dir.glob(ext))
    return [f for f in data_files if '标注' not in f.name and not f.name.endswith('_predicted.json')]

def calculate_metrics(y_true, y_pred):
    """
    计算分类指标 (Precision, Recall, F1)
    y_true, y_pred: [N, 9] 形状的矩阵
    """
    # 转换为二进制（伤损 vs 正常）
    true_mask = (y_true > 0)
    pred_mask = (y_pred > 0)
    
    tp = np.sum(true_mask & pred_mask)
    fp = np.sum((~true_mask) & pred_mask)
    fn = np.sum(true_mask & (~pred_mask))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_true': np.sum(true_mask),
        'total_pred': np.sum(pred_mask)
    }

# ======================================
# 主流程
# ======================================
def run_full_test():
    print("="*60)
    print(f"HDC 点级分类器完整测试流程 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")

    # 1. 环境准备与数据转换
    clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR])
    
    print("\n[Step 1] 正在进行数据转换 (JSON -> CSV)...")
    converter = UltrasonicDataConverter()
    for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
        json_files = get_json_files(data_dir)
        for json_file in json_files:
            annotation_file = json_file.parent / f"{json_file.stem}标注{json_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            try:
                converter.generate_all_csvs(str(json_file), annotation_path)
                # print(f"  ✓ {json_file.name}")
            except Exception as e:
                print(f"  ✗ {json_file.name} 转换失败: {e}")

    # 2. 模型检查与加载
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] 模型文件不存在: {MODEL_PATH}")
        print("请先运行 hdc_point_level_classifier.py 进行训练。")
        return

    print("\n[Step 2] 加载 HDC 模型...")
    hdc_sys = HDCPointLevelModel(device=device)
    hdc_sys.load(str(MODEL_PATH))

    # 3. 执行预测
    print("\n[Step 3] 正在对测试集进行点级预测...")
    test_ds = HDCPointLevelDataset(TEST_DATA_DIR, is_train=False)
    if len(test_ds) == 0:
        print("[WARN] 测试集为空，跳过预测。")
    else:
        test_ld = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
        predictions = hdc_sys.predict(test_ld) # [N, 9]
        
        # 将预测结果分发到各个文件并计算指标
        print("\n[Step 4] 计算性能指标...")
        
        file_metrics = {}
        all_y_true = []
        all_y_pred = []
        
        # 记录每个文件的预测结果
        results_by_file = {}
        for idx, pred_row in enumerate(predictions):
            info = test_ds.file_info[idx]
            file_name = Path(info['file']).name
            if file_name not in results_by_file:
                results_by_file[file_name] = []
            results_by_file[file_name].append({
                'encoding': info['center_encoding'],
                'pred': pred_row
            })

        for file_name, results in results_by_file.items():
            # 获取真实标签
            base_name = file_name.replace("_hdc.csv", "")
            label_file = TEST_DATA_DIR / f"{base_name}_label.csv"
            
            if not label_file.exists():
                print(f"  [SKIP] {file_name} 无标签文件，无法计算指标。")
                continue
                
            # 读取特征和标签
            df_hdc = pd.read_csv(TEST_DATA_DIR / file_name)
            df_label = pd.read_csv(label_file)
            df_merged = pd.merge(df_hdc[['encoding']], df_label, on='encoding', how='left').fillna(0)
            
            # 准备预测矩阵
            y_pred_matrix = np.zeros((len(df_merged), NUM_CHANNELS), dtype=np.int64)
            y_true_matrix = df_merged[[f'label_{i}' for i in range(NUM_CHANNELS)]].values
            
            # 填充预测值
            encoding_to_idx = {enc: i for i, enc in enumerate(df_merged['encoding'].values)}
            for res in results:
                if res['encoding'] in encoding_to_idx:
                    idx = encoding_to_idx[res['encoding']]
                    y_pred_matrix[idx] = res['pred']
            
            # 保存预测结果 CSV
            pred_df = pd.DataFrame({'encoding': df_merged['encoding']})
            for i in range(NUM_CHANNELS):
                pred_df[f'predicted_label_{i}'] = y_pred_matrix[:, i]
            
            out_csv = TEST_DATA_DIR / f"{base_name}_predicted_label.csv"
            pred_df.to_csv(out_csv, index=False)
            
            # 计算该文件的指标
            metrics = calculate_metrics(y_true_matrix, y_pred_matrix)
            file_metrics[file_name] = metrics
            
            all_y_true.append(y_true_matrix)
            all_y_pred.append(y_pred_matrix)
            
            print(f"  文件: {file_name}")
            print(f"    Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"    真缺陷点数: {metrics['total_true']}, 预测点数: {metrics['total_pred']}, 正确识别: {metrics['tp']}")

        # 计算总体指标
        if all_y_true:
            final_y_true = np.vstack(all_y_true)
            final_y_pred = np.vstack(all_y_pred)
            total_metrics = calculate_metrics(final_y_true, final_y_pred)
            
            print("\n" + "="*40)
            print("汇总评估报告")
            print("="*40)
            print(f"总体精准率 (Precision): {total_metrics['precision']:.4f}")
            print(f"总体召回率 (Recall):    {total_metrics['recall']:.4f}")
            print(f"总体 F1 分数 (F1-Score): {total_metrics['f1']:.4f}")
            print("-" * 40)
            print(f"总真缺陷点数: {total_metrics['total_true']}")
            print(f"总预测缺陷点: {total_metrics['total_pred']}")
            print(f"总正确识别点: {total_metrics['tp']}")
            print(f"总漏检点数 (FN): {total_metrics['fn']}")
            print(f"总误报点数 (FP): {total_metrics['fp']}")
            print("="*40)
            
            # 保存结果到文件
            report_path = RESULTS_DIR / f"hdc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("HDC 点级分类器评估报告\n")
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {MODEL_PATH}\n")
                f.write("="*30 + "\n")
                f.write(f"Precision: {total_metrics['precision']:.4f}\n")
                f.write(f"Recall:    {total_metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {total_metrics['f1']:.4f}\n")
                f.write(f"Total True: {total_metrics['total_true']}\n")
                f.write(f"Total Pred: {total_metrics['total_pred']}\n")
                f.write(f"Total Correct: {total_metrics['tp']}\n")
            print(f"\n[INFO] 详细报告已保存至: {report_path}")

if __name__ == "__main__":
    run_full_test()
