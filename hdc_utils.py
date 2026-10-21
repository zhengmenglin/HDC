#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDC 公共工具模块
包含两个测试脚本共用的工具函数和常量
"""

import os
import json
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

FLAW_ID_TO_NAME = {
    1: "轨头外侧核伤", 2: "轨头中心核伤", 3: "轨头内侧核伤", 4: "轨头内、外、中心核伤",
    5: "轨头内、中心核伤", 6: "轨头外、中心核伤", 7: "轨头水平分裂", 8: "轨头、轨腰分裂",
    9: "轨腰水平裂", 10: "轨底断裂", 11: "螺孔裂纹", 12: "接头", 13: "螺孔",
    14: "导线孔", 15: "焊缝", 16: "鱼鳞伤", 17: "轨底锈蚀", 18: "轨底焊筋", 19: "道岔"
}

CLEAN_FILE_PATTERNS = {
    "train": ["*_encoding.csv"],
    "predict": ["*_predicted_label.csv", "*_predicted.json"],
    "all": ["*_hdc.csv", "*_label.csv", "*_encoding.csv", "*_predicted_label.csv", "*_predicted.json"]
}


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_learned_files(learned_files_path):
    """加载已学习的文件列表"""
    if learned_files_path.exists():
        try:
            with open(learned_files_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            print(f"[WARN] 加载已学习文件列表失败: {e}")
    return set()


def save_learned_files(learned_files, learned_files_path):
    """保存已学习的文件列表"""
    try:
        learned_files_path.parent.mkdir(parents=True, exist_ok=True)
        with open(learned_files_path, 'w', encoding='utf-8') as f:
            json.dump(list(learned_files), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存已学习文件列表失败: {e}")


def clean_old_files(data_dirs, file_types="all"):
    """清理旧的中间文件，保持环境整洁"""
    patterns = CLEAN_FILE_PATTERNS.get(file_types, CLEAN_FILE_PATTERNS["all"])
    
    if file_types != "all":
        print(f"\n[INFO] 正在清理旧的{file_types}相关文件...")
    else:
        print("\n[INFO] 正在清理所有旧文件...")
    
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
    """获取目录下的原始 JSON 数据文件（排除标注/派生文件）。

    注意：*.json 也会匹配 *.xxc.json，旧实现会把同一个文件返回两次，
    导致训练样本被重复计入（实测 1855 → 3710）。这里按绝对路径去重。
    """
    if not data_dir.exists():
        return []
    candidates = []
    for ext in ['*.json', '*.xxc.json']:
        candidates.extend(data_dir.glob(ext))
    out, seen = [], set()
    for f in candidates:
        key = f.resolve()
        if key in seen:
            continue
        if ('标注' in f.name or f.name.endswith('_predicted.json')
                or f.name.endswith('_hdc.json')):
            continue
        seen.add(key)
        out.append(f)
    return out


def calculate_metrics(y_true, y_pred):
    """
    计算分类指标 (Precision, Recall, F1)
    y_true, y_pred: [N, C] 形状的矩阵，C为通道数或1
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
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'total_true': int(np.sum(true_mask)), 'total_pred': int(np.sum(pred_mask))
    }


def convert_predicted_csv_to_json(csv_path, original_json_path, output_json_path, hdc_csv_path=None, channel_ids=None):
    """
    将预测结果 CSV 转换回 GUI 可识别的 _predicted.json 格式
    
    参数:
    - csv_path: 预测结果CSV路径
    - original_json_path: 原始JSON文件路径
    - output_json_path: 输出JSON路径
    - hdc_csv_path: HDC CSV文件路径（用于提取深度）
    - channel_ids: 要处理的通道ID列表，None表示处理所有通道
    """
    if not os.path.exists(csv_path) or not os.path.exists(original_json_path):
        return

    beam_offset_map = {}
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
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

    df_pred = pd.read_csv(csv_path)
    depth_map = {}
    if hdc_csv_path and os.path.exists(hdc_csv_path):
        try:
            df_hdc = pd.read_csv(hdc_csv_path)
            for _, row in df_hdc.iterrows():
                enc = int(row['encoding'])
                if channel_ids is not None:
                    for ch in channel_ids:
                        depth_map[(enc, ch)] = row[f'channel_{ch}']
                else:
                    for ch in range(9):
                        depth_map[(enc, ch)] = row[f'channel_{ch}']
        except Exception as e:
            print(f"  [WARN] 读取 {hdc_csv_path} 失败: {e}")

    annotations_groups = {}
    
    if channel_ids is None:
        channel_ids = range(9)
    
    for _, row in df_pred.iterrows():
        encoding = int(row['encoding'])
        for ch in channel_ids:
            label_col = f'predicted_label_{ch}'
            if label_col in row and row[label_col] > 0:
                label_id = int(row[label_col])
                offset = beam_offset_map.get(ch, 0)
                original_e = int(encoding + offset)
                
                if channel_ids is not None and len(channel_ids) == 1:
                    depth = depth_map.get(encoding, 0.0)
                else:
                    depth = depth_map.get((encoding, ch), 0.0)
                
                if label_id not in annotations_groups:
                    annotations_groups[label_id] = []
                
                annotations_groups[label_id].append({
                    "e": original_e,
                    "channel": ch,
                    "depth": float(depth),
                    "amplitude": 100.0
                })
    
    annotations_list = []
    annotation_id = 1000
    
    for label_id, points in annotations_groups.items():
        if not points: continue
        
        name = FLAW_ID_TO_NAME.get(label_id, "未知伤损")
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


def load_csv_with_labels(csv_file, data_dir, num_channels=9):
    """
    加载CSV数据并合并标签
    
    参数:
    - csv_file: CSV文件路径
    - data_dir: 数据目录
    - num_channels: 通道数量
    
    返回:
    - df: 合并后的DataFrame
    - channel_cols: 通道列名列表
    - label_cols: 标签列名列表
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取文件 {csv_file} 失败: {e}")
        return None, None, None
    
    base_name = csv_file.name.replace("_hdc.csv", "")
    
    label_file = data_dir / f"{base_name}_label.csv"
    channel_cols = [f'channel_{i}' for i in range(num_channels)]
    label_cols = [f'label_{i}' for i in range(num_channels)]
    
    if label_file.exists():
        try:
            label_df = pd.read_csv(label_file)
            if num_channels == 1:
                if 'label_8' in label_df.columns:
                    df = pd.merge(df, label_df[["encoding", "label_8"]], on='encoding', how='left').fillna(0)
                else:
                    df['label_8'] = 0
            else:
                if all(f'label_{i}' in label_df.columns for i in range(num_channels)):
                    df = pd.merge(df, label_df, on='encoding', how='left').fillna(0)
                else:
                    for i in range(num_channels):
                        df[f'label_{i}'] = 0
        except Exception as e:
            print(f"读取或合并标签文件 {label_file} 失败: {e}")
            for i in range(num_channels):
                df[f'label_{i}'] = 0
    else:
        for i in range(num_channels):
            df[f'label_{i}'] = 0
    
    if not all(col in df.columns for col in channel_cols):
        print(f"文件 {csv_file.name} 缺少必要的通道列，已跳过")
        return None, None, None
    
    for col in label_cols:
        if col not in df.columns:
            df[col] = 0
    
    df = df.sort_values(by='encoding')
    
    return df, channel_cols, label_cols


def get_channel_info(channels):
    """
    获取通道配置信息
    
    参数:
    - channels: 通道列表，如 [0,1,2,3,4,5,6,7,8] 或 [8]
    
    返回:
    - num_channels: 通道数量
    - is_single_channel: 是否单通道模式
    - channel_id: 单通道模式下的通道ID
    """
    num_channels = len(channels)
    is_single_channel = num_channels == 1
    channel_id = channels[0] if is_single_channel else None
    return num_channels, is_single_channel, channel_id


def process_prediction_results(predictions, normal_sims, max_defect_sims, all_class_sims, 
                                file_info, df_hdc, channels, file_name=None):
    """
    处理预测结果，填充预测矩阵
    
    参数:
    - predictions: 预测结果数组 [N, num_channels]
    - normal_sims: 正常类相似度数组 [N, num_channels]
    - max_defect_sims: 最大伤损类相似度数组 [N, num_channels]
    - all_class_sims: 所有类别相似度数组 [N, num_channels, num_classes]
    - file_info: 文件信息列表
    - df_hdc: HDC数据DataFrame
    - channels: 通道列表
    
    返回:
    - y_pred_matrix: 预测矩阵
    - y_normal_sim_matrix: 正常类相似度矩阵
    - y_max_defect_sim_matrix: 最大伤损类相似度矩阵
    - y_class_sims_dict: 特定类别相似度字典 {class_id: similarity_matrix}
    """
    num_channels = len(channels)
    is_single_channel = num_channels == 1
    
    y_pred_matrix = np.zeros((len(df_hdc), num_channels), dtype=np.int64)
    y_normal_sim_matrix = np.zeros((len(df_hdc), num_channels), dtype=np.float32)
    y_max_defect_sim_matrix = np.zeros((len(df_hdc), num_channels), dtype=np.float32)
    
    y_class_0_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
    y_class_11_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
    y_class_13_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
    
    encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
    
    for idx, pred_row in enumerate(predictions):
        info = file_info[idx]
        # 旧版用全局 predictions 填单文件矩阵：不同文件的 encoding 重叠时，
        # 别的文件的预测会覆盖进来（跨文件串台）。这里按文件名过滤。
        if file_name is not None and Path(info['file']).name != file_name:
            continue
        if info['center_encoding'] in encoding_to_idx:
            matrix_idx = encoding_to_idx[info['center_encoding']]
            y_pred_matrix[matrix_idx] = pred_row
            y_normal_sim_matrix[matrix_idx] = normal_sims[idx]
            y_max_defect_sim_matrix[matrix_idx] = max_defect_sims[idx]
            
            if is_single_channel:
                channel_sims = all_class_sims[idx][0] if len(all_class_sims[idx]) > 0 else []
            else:
                target_ch = 8 if 8 in channels else channels[-1]
                ch_idx = channels.index(target_ch) if target_ch in channels else 0
                channel_sims = all_class_sims[idx][ch_idx] if len(all_class_sims[idx]) > ch_idx else []
            
            if len(channel_sims) > 0:
                y_class_0_sim_matrix[matrix_idx] = channel_sims[0]
            if len(channel_sims) > 11:
                y_class_11_sim_matrix[matrix_idx] = channel_sims[11]
            if len(channel_sims) > 13:
                y_class_13_sim_matrix[matrix_idx] = channel_sims[13]
    
    return (y_pred_matrix, y_normal_sim_matrix, y_max_defect_sim_matrix,
            {0: y_class_0_sim_matrix, 11: y_class_11_sim_matrix, 13: y_class_13_sim_matrix})


def save_prediction_csv(df_hdc, y_pred_matrix, y_normal_sim_matrix, y_max_defect_sim_matrix,
                        y_class_sims_dict, channels, output_path):
    """
    保存预测结果CSV
    
    参数:
    - df_hdc: HDC数据DataFrame
    - y_pred_matrix: 预测矩阵
    - y_normal_sim_matrix: 正常类相似度矩阵
    - y_max_defect_sim_matrix: 最大伤损类相似度矩阵
    - y_class_sims_dict: 特定类别相似度字典
    - channels: 通道列表
    - output_path: 输出路径
    """
    pred_df = pd.DataFrame({'encoding': df_hdc['encoding']})
    
    for i, ch in enumerate(channels):
        pred_df[f'predicted_label_{ch}'] = y_pred_matrix[:, i]
        sim_strs = [f"{n:.4f},{d:.4f}" for n, d in zip(y_normal_sim_matrix[:, i], y_max_defect_sim_matrix[:, i])]
        pred_df[f'similarity_{ch}'] = sim_strs
    
    pred_df['class_0_similarity'] = y_class_sims_dict[0][:, 0]
    pred_df['class_11_similarity'] = y_class_sims_dict[11][:, 0]
    pred_df['class_13_similarity'] = y_class_sims_dict[13][:, 0]
    
    pred_df.to_csv(output_path, index=False)


def generate_report(total_metrics, model_path, results_dir, channels, timestamp=None,
                    split_name="训练集"):
    """
    生成评估报告
    
    参数:
    - total_metrics: 指标字典
    - model_path: 模型路径
    - results_dir: 结果目录
    - channels: 通道列表
    - timestamp: 时间戳
    
    返回:
    - report_path: 报告路径
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    channel_str = "通道8" if len(channels) == 1 and channels[0] == 8 else f"{len(channels)}通道"
    
    print("\n" + "="*40)
    print(f"{split_name}汇总评估报告 ({channel_str})")
    print("="*40)
    print(f"总体精准率 (Precision): {total_metrics['precision']:.4f}")
    print(f"总体召回率 (Recall):    {total_metrics['recall']:.4f}")
    print(f"总体 F1 分数 (F1-Score): {total_metrics['f1']:.4f}")
    print("-" * 40)
    print(f"总真缺陷点数: {total_metrics['total_true']}")
    print(f"总预测缺陷点: {total_metrics['total_pred']}")
    print(f"总正确识别点: {total_metrics['tp']}")
    print("="*40)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = "test" if split_name == "测试集" else "train"
    report_path = results_dir / f"hdc_{tag}_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"HDC {split_name}评估报告 ({channel_str})\n")
        f.write(f"测试时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write("="*30 + "\n")
        f.write(f"Precision: {total_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {total_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {total_metrics['f1']:.4f}\n")
        f.write(f"Total True: {total_metrics['total_true']}\n")
        f.write(f"Total Pred: {total_metrics['total_pred']}\n")
        f.write(f"Total Correct: {total_metrics['tp']}\n")
    
    print(f"\n[INFO] 详细报告已保存至: {report_path}")
    return report_path
