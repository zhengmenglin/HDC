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
import json

# 导入数据转换器
from ultrasonic_data_converter import UltrasonicDataConverter

# 反向映射：从 ID 到名称
FLAW_ID_TO_NAME = {
    1: "轨头外侧核伤", 2: "轨头中心核伤", 3: "轨头内侧核伤", 4: "轨头内、外、中心核伤",
    5: "轨头内、中心核伤", 6: "轨头外、中心核伤", 7: "轨头水平分裂", 8: "轨头、轨腰分裂",
    9: "轨腰水平裂", 10: "轨底断裂", 11: "螺孔裂纹", 12: "接头", 13: "螺孔",
    14: "导线孔", 15: "焊缝", 16: "鱼鳞伤", 17: "轨底锈蚀", 18: "轨底焊筋", 19: "道岔"
}

# ======================================
# 配置参数
# ======================================
TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
TEST_DATA_DIR = Path(r"d:\HDC\data\test")
MODEL_PATH = Path(r"d:\HDC\models\hdc_point_level_9ch.pth")
LEARNED_FILES_PATH = Path(r"d:\HDC\models\hdc_point_level_9ch.json")
RESULTS_DIR = Path(r"d:\HDC\hdc_test_results")

# ======================================
# 工具函数
# ======================================
def load_learned_files():
    """加载已学习的文件列表"""
    if LEARNED_FILES_PATH.exists():
        try:
            with open(LEARNED_FILES_PATH, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"[WARN] 加载已学习文件列表失败: {e}")
    return set()

def save_learned_files(learned_files):
    """保存已学习的文件列表"""
    try:
        LEARNED_FILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEARNED_FILES_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(learned_files), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存已学习文件列表失败: {e}")
def convert_predicted_csv_to_json(csv_path, original_json_path, output_json_path, hdc_csv_path=None):
    """
    将预测结果 CSV 转换回 GUI 可识别的 _predicted.json 格式
    """
    if not os.path.exists(csv_path) or not os.path.exists(original_json_path):
        return

    # 1. 获取 BeamOffset
    beam_offset_map = {}
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 处理可能是列表或字典的情况
            if isinstance(data, list) and len(data) > 0 and 'BeamOffset' in data[0]:
                bo = data[0]['BeamOffset']
            elif isinstance(data, dict) and 'BeamOffset' in data:
                bo = data['BeamOffset']
            else:
                bo = []
            
            for i in range(0, len(bo), 2):
                beam_offset_map[int(bo[i])] = bo[i+1]
    except Exception as e:
        print(f"  [WARN] 读取 {original_json_path} 的 BeamOffset 失败: {e}")

    # 2. 读取 CSV 预测结果 和 原始数据(用于提取深度)
    df_pred = pd.read_csv(csv_path)
    df_hdc = None
    depth_map = {}
    if hdc_csv_path and os.path.exists(hdc_csv_path):
        try:
            df_hdc = pd.read_csv(hdc_csv_path)
            # 建立 (encoding, channel) -> depth 的映射
            for _, row in df_hdc.iterrows():
                enc = int(row['encoding'])
                for ch in range(9):
                    depth_map[(enc, ch)] = row[f'channel_{ch}']
        except Exception as e:
            print(f"  [WARN] 读取 {hdc_csv_path} 失败: {e}")

    # 3. 按标签分组收集点
    # 格式: { label_id: [ {e, channel, depth, amplitude}, ... ] }
    annotations_groups = {}
    
    for _, row in df_pred.iterrows():
        encoding = int(row['encoding'])
        for ch in range(9):
            label_col = f'predicted_label_{ch}'
            if label_col in row and row[label_col] > 0:
                label_id = int(row[label_col])
                offset = beam_offset_map.get(ch, 0)
                original_e = int(encoding + offset)
                
                depth = depth_map.get((encoding, ch), 0.0)
                
                if label_id not in annotations_groups:
                    annotations_groups[label_id] = []
                
                annotations_groups[label_id].append({
                    "e": original_e,
                    "channel": ch,
                    "depth": float(depth),
                    "amplitude": 100.0  # 默认 100.0
                })
    
    # 4. 构建 JSON 结构
    annotations_list = []
    annotation_id = 1000
    
    for label_id, points in annotations_groups.items():
        if not points: continue
        
        name = FLAW_ID_TO_NAME.get(label_id, "未知伤损")
        
        # 计算 start_E, end_E, channels
        es = [p['e'] for p in points]
        chs = sorted(list(set([p['channel'] for p in points])))
        
        annotations_list.append({
            "id": str(annotation_id),
            "type": "伤损" if label_id < 12 else "结构",
            "subtype": {"name": name, "id": str(label_id)},
            "points": points,
            "start_E": min(es),
            "end_E": max(es),
            "channels": chs,
            "timestamp": datetime.now().isoformat(),
            "source": "predicted"
        })
        annotation_id += 1
    
    result = {"annotations": annotations_list}
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # print(f"  ✓ 已生成预测标注: {os.path.basename(output_json_path)}")

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
    return [f for f in data_files if '标注' not in f.name and not f.name.endswith('_predicted.json') and not f.name.endswith('_hdc.json')]

def calculate_metrics(y_true, y_pred):
    """
    计算分类指标 (Precision, Recall, F1)
    y_true, y_pred: [N, 9] 形状的矩阵
    """
    true_mask = (y_true > 0)
    pred_mask = (y_pred > 0)
    
    tp = np.sum(true_mask & pred_mask)
    fp = np.sum((~true_mask) & pred_mask)
    fn = np.sum(true_mask & (~pred_mask))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'total_true': np.sum(true_mask), 'total_pred': np.sum(pred_mask)
    }

def filter_noise_predictions(y_pred_matrix, min_neighbors=5, window_size=11):
    """
    增强版噪声过滤。
    y_pred_matrix: [N, 9] 形状的矩阵
    """
    mask = (y_pred_matrix > 0).astype(np.int32)
    filtered_mask = np.zeros_like(mask)
    for ch in range(mask.shape[1]):
        ch_mask = mask[:, ch]
        kernel = np.ones(window_size)
        neighbor_counts = np.convolve(ch_mask, kernel, mode='same')
        # 提高判定门槛，必须在周围 window_size 内有足够多的邻居
        filtered_mask[:, ch] = (ch_mask == 1) & (neighbor_counts >= min_neighbors)
    return y_pred_matrix * filtered_mask

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
    # 不再盲目清理旧文件，而是根据增量逻辑判断
    # clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR])
    
    learned_files = load_learned_files()
    new_trained_files = []
    
    print("\n[Step 1] 正在进行数据转换 (JSON -> CSV)...")
    converter = UltrasonicDataConverter()
    for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
        json_files = get_json_files(data_dir)
        for json_file in json_files:
            file_name = json_file.name
            # 如果是训练集且已学习，跳过转换（除非 CSV 不存在）
            hdc_csv = json_file.parent / f"{json_file.stem}_hdc.csv"
            if data_dir == TRAIN_DATA_DIR and file_name in learned_files and hdc_csv.exists():
                continue
                
            annotation_file = json_file.parent / f"{json_file.stem}标注{json_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            try:
                converter.generate_all_csvs(str(json_file), annotation_path)
                if data_dir == TRAIN_DATA_DIR:
                    new_trained_files.append(file_name)
                # print(f"  ✓ {json_file.name}")
            except Exception as e:
                print(f"  ✗ {json_file.name} 转换失败: {e}")

    # 2. 模型训练 (增量)
    print("\n[Step 2] 正在进行 HDC 模型训练...")
    hdc_sys = HDCPointLevelModel(device=device)
    
    # 如果已有模型，先加载
    if MODEL_PATH.exists():
        print(f"[INFO] 加载现有模型进行增量训练: {MODEL_PATH}")
        hdc_sys.load(str(MODEL_PATH))
    
    # 筛选出真正需要训练的新文件
    actual_new_files = [f for f in get_json_files(TRAIN_DATA_DIR) if f.name not in learned_files]
    
    if not actual_new_files:
        print("[INFO] 没有新数据需要训练。")
    else:
        print(f"[INFO] 发现 {len(actual_new_files)} 个新文件，开始增量学习...")
        # 准备新文件的路径列表
        new_csv_files = [TRAIN_DATA_DIR / f"{Path(f).stem}_hdc.csv" for f in actual_new_files]
        # 创建一个临时的 Dataset 只包含新文件
        train_ds = HDCPointLevelDataset(TRAIN_DATA_DIR, is_train=True, files=new_csv_files)
        
        if len(train_ds) == 0:
            print("[WARN] 过滤后训练集为空。")
        else:
            print(f"[INFO] 训练样本数: {len(train_ds)}")
            batch_size = 32 if torch.cuda.is_available() else 8
            train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            hdc_sys.train(train_ld)
            
            # 保存模型
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            hdc_sys.save(str(MODEL_PATH))
            
            # 更新已学习列表
            for f in actual_new_files:
                learned_files.add(f.name)
            save_learned_files(learned_files)
            print(f"[INFO] 已更新学习记录: {len(learned_files)} 个文件")

    # 4. 执行预测与转换
    print("\n[Step 4] 正在对数据集进行点级预测与转换...")
    
    # 定义处理流程：(目录, 是否计算指标)
    tasks = [
        (TRAIN_DATA_DIR, True, "训练集"),
        (TEST_DATA_DIR, False, "测试集")
    ]
    
    all_train_y_true = []
    all_train_y_pred = []

    # 设置预测 batch size
    eval_batch_size = 32 if torch.cuda.is_available() else 4
    print(f"[INFO] 预测 Batch Size: {eval_batch_size}")

    for data_dir, calc_metrics, label in tasks:
        print(f"\n--- 处理{label} ({data_dir}) ---")
        ds = HDCPointLevelDataset(data_dir, is_train=False) # 预测时不需要 shuffle
        if len(ds) == 0:
            print(f"[WARN] {label}为空，跳过。")
            continue
            
        ld = torch.utils.data.DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
        # 使用 Margin 逻辑预测：只有当“伤损类”相似度显著高于“正常类”时才保留
        # 基础阈值设低(0.1)，主要靠 Margin(0.05) 过滤误报
        predictions, similarities = hdc_sys.predict(ld, threshold=0.95, margin=0.5) 
        
        # 记录每个文件的预测结果
        results_by_file = {}
        for idx, pred_row in enumerate(predictions):
            info = ds.file_info[idx]
            file_name = Path(info['file']).name
            if file_name not in results_by_file:
                results_by_file[file_name] = []
            results_by_file[file_name].append({
                'encoding': info['center_encoding'],
                'pred': pred_row,
                'sim': similarities[idx]
            })

        for file_name, results in results_by_file.items():
            base_name = file_name.replace("_hdc.csv", "")
            hdc_file = data_dir / file_name
            label_file = data_dir / f"{base_name}_label.csv"
            
            # 读取特征
            df_hdc = pd.read_csv(hdc_file)
            
            # 准备预测矩阵
            y_pred_matrix = np.zeros((len(df_hdc), NUM_CHANNELS), dtype=np.int64)
            y_sim_matrix = np.zeros((len(df_hdc), NUM_CHANNELS), dtype=np.float32)
            
            # 填充预测值
            encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
            for res in results:
                if res['encoding'] in encoding_to_idx:
                    idx = encoding_to_idx[res['encoding']]
                    y_pred_matrix[idx] = res['pred']
                    y_sim_matrix[idx] = res['sim']
            
            # --- 应用噪声过滤 ---
            # 注意：噪声过滤可能改变预测值，但我们暂时保留过滤前的相似度
            y_pred_matrix = filter_noise_predictions(y_pred_matrix)
            
            # 保存预测结果 CSV
            pred_df = pd.DataFrame({'encoding': df_hdc['encoding']})
            for i in range(NUM_CHANNELS):
                pred_df[f'predicted_label_{i}'] = y_pred_matrix[:, i]
                pred_df[f'similarity_{i}'] = y_sim_matrix[:, i]
            
            pred_csv_path = data_dir / f"{base_name}_predicted_label.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            
            # --- 新增：转换为 JSON ---
            # 寻找原始 JSON 文件
            original_json = None
            for ext in ['.json', '.xxc.json']:
                potential_json = data_dir / f"{base_name}{ext}"
                if potential_json.exists():
                    original_json = potential_json
                    break
            
            if original_json:
                output_json = data_dir / f"{base_name}_predicted.json"
                convert_predicted_csv_to_json(pred_csv_path, original_json, output_json, hdc_file)
                # print(f"  ✓ {base_name}_predicted.json")

            # --- 计算指标 (仅限训练集) ---
            if calc_metrics and label_file.exists():
                df_label = pd.read_csv(label_file)
                df_merged = pd.merge(df_hdc[['encoding']], df_label, on='encoding', how='left').fillna(0)
                y_true_matrix = df_merged[[f'label_{i}' for i in range(NUM_CHANNELS)]].values
                
                metrics = calculate_metrics(y_true_matrix, y_pred_matrix)
                all_train_y_true.append(y_true_matrix)
                all_train_y_pred.append(y_pred_matrix)
                
                print(f"  文件: {file_name} -> Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

    # 5. 汇总训练集指标
    if all_train_y_true:
        final_y_true = np.vstack(all_train_y_true)
        final_y_pred = np.vstack(all_train_y_pred)
        total_metrics = calculate_metrics(final_y_true, final_y_pred)
        
        print("\n" + "="*40)
        print("训练集汇总评估报告")
        print("="*40)
        print(f"总体精准率 (Precision): {total_metrics['precision']:.4f}")
        print(f"总体召回率 (Recall):    {total_metrics['recall']:.4f}")
        print(f"总体 F1 分数 (F1-Score): {total_metrics['f1']:.4f}")
        print("-" * 40)
        print(f"总真缺陷点数: {total_metrics['total_true']}")
        print(f"总预测缺陷点: {total_metrics['total_pred']}")
        print(f"总正确识别点: {total_metrics['tp']}")
        print("="*40)
        
        # 保存结果到文件
        report_path = RESULTS_DIR / f"hdc_train_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HDC 训练集评估报告\n")
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
