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
import random
import shutil
import matplotlib.pyplot as plt

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

# 导入配置管理
from config import config

# ======================================
# 设置随机种子以确保结果可复现
# ======================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

from sklearn.model_selection import KFold

# 反向映射：从 ID 到名称
FLAW_ID_TO_NAME = {
    1: "轨头外侧核伤", 2: "轨头中心核伤", 3: "轨头内侧核伤", 4: "轨头内、外、中心核伤",
    5: "轨头内、中心核伤", 6: "轨头外、中心核伤", 7: "轨头水平分裂", 8: "轨头、轨腰分裂",
    9: "轨腰水平裂", 10: "轨底断裂", 11: "螺孔裂纹", 12: "接头", 13: "螺孔",
    14: "导线孔", 15: "焊缝", 16: "鱼鳞伤", 17: "轨底锈蚀", 18: "轨底焊筋", 19: "道岔"
}

# ======================================
# 从配置文件加载参数
# ======================================
TRAIN_DATA_DIR = Path(config.get("data.train_dir"))
TEST_DATA_DIR = Path(config.get("data.test_dir"))
MODEL_PATH = Path(config.get("model.model_path"))
LEARNED_FILES_PATH = Path(config.get("model.learned_files_path"))
RESULTS_DIR = Path(config.get("test.results_dir"))

# 学习方式配置
LEARNING_METHOD = config.get("model.learning_method", "onlinehd")
RIDGE_ALPHA = config.get("model.ridge_alpha", 1.0)
LEARNING_RATE = config.get("model.learning_rate", 0.1)
EPOCHS = config.get("model.epochs", 5)

# 测试配置
TRAIN_STRATEGY = config.get("test.train_strategy", "damage_only")
PREDICT_THRESHOLD = config.get("test.predict_threshold", 0)
PREDICT_MARGIN = config.get("test.predict_margin", 0)

# ======================================
# 从hdc_point_level_classifier.py导入常量
# ======================================
from hdc_point_level_classifier import CENTER_IDX

# ======================================
# 扩展数据集类
# ======================================
class HDCCenterDamageDataset(HDCPointLevelDataset):
    """
    只包含窗口中心点为伤损的样本数据集类
    过滤掉窗口中心点标签全为0的正常样本
    """
    def _load_data(self):
        """
        加载 CSV 数据并生成滑动窗口样本，只保留中心点为伤损的样本。
        """
        if self.specified_files is not None:
            csv_files = [Path(f) for f in self.specified_files]
        else:
            csv_files = list(self.data_dir.glob("*_hdc.csv"))
            
        if not csv_files:
            print(f"警告: 在 {self.data_dir} 未找到 *_hdc.csv 文件")
            return
        
        self.samples.clear()
        self.labels.clear()
        self.file_info.clear()
        
        total_windows = 0
        damage_windows = 0
        
        for csv_file in csv_files:
            # 1. 读取特征数据 (1_hdc.csv)
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"读取文件 {csv_file} 失败: {e}")
                continue
        
            base_name = csv_file.name.replace("_hdc.csv", "")
            
            # 2. 读取标签数据 (1_label.csv)
            label_file = self.data_dir / f"{base_name}_label.csv"
            if label_file.exists():
                try:
                    label_df = pd.read_csv(label_file)
                    # 确保标签列存在
                    if all(f'label_{i}' in label_df.columns for i in range(NUM_CHANNELS)):
                        # 按照 encoding 合并特征和标签
                        df = pd.merge(df, label_df, on='encoding', how='left').fillna(0)
                    else:
                        # 如果标签文件格式不对，填充 0
                        for i in range(NUM_CHANNELS):
                            df[f'label_{i}'] = 0
                except Exception as e:
                    print(f"读取或合并标签文件 {label_file} 失败: {e}")
                    # 回退到全 0 标签
                    for i in range(NUM_CHANNELS):
                        df[f'label_{i}'] = 0
            else:
                # 如果没有标签文件，所有通道标签设为 0（正常）
                for i in range(NUM_CHANNELS):
                    df[f'label_{i}'] = 0
            
            # 3. 确保所有必要的列都存在
            channel_cols = [f'channel_{i}' for i in range(NUM_CHANNELS)]
            label_cols = [f'label_{i}' for i in range(NUM_CHANNELS)]
            
            if not all(col in df.columns for col in channel_cols):
                print(f"文件 {csv_file.name} 缺少必要的通道列，已跳过")
                continue
            
            if not all(col in df.columns for col in label_cols):
                # 这种情况不应该发生，因为上面已经做了填充，但为了安全再次检查
                for col in label_cols:
                    if col not in df.columns:
                        df[col] = 0
            
            # 按照编码值排序，确保时序/空间连续性
            df = df.sort_values(by='encoding')
        
            channel_data = df[channel_cols].values.astype(np.float32)
            label_data = df[label_cols].values.astype(np.int64)
            encodings = df['encoding'].values
        
            # 4. 滑动窗口采样，只保留中心点为伤损的样本
            if len(df) < self.window_size:
                print(f"跳过文件 {csv_file.name}: 数据行数 ({len(df)}) 小于窗口大小 ({self.window_size})")
                continue
        
            # 步长为 1，确保每个点都能作为中心点被预测
            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                window_x = channel_data[start:end]
                
                # 获取中心行的 9 个通道标签 (index 63)
                center_labels = label_data[start + CENTER_IDX]
                
                total_windows += 1
                
                # 只保留中心点标签不为0的样本（即存在伤损）
                if any(center_labels > 0):
                    damage_windows += 1
                    self.samples.append(window_x)
                    self.labels.append(center_labels)
                    
                    if not self.is_train:
                        self.file_info.append({
                            'file': str(csv_file),
                            'center_encoding': encodings[start + CENTER_IDX],
                            'start': start
                        })
        
        print(f"数据集加载完成: {len(self.samples)} 个伤损样本 (总窗口数: {total_windows}, 伤损窗口占比: {damage_windows/total_windows*100:.2f}%)")


class HDCMixedDamageDataset(HDCPointLevelDataset):
    """
    混合伤损样本和正常样本的数据集类
    包含所有伤损样本，以及与伤损样本数量相同的正常样本
    """
    def _load_data(self):
        """
        加载 CSV 数据并生成滑动窗口样本，包含伤损样本和同等数量的正常样本。
        """
        if self.specified_files is not None:
            csv_files = [Path(f) for f in self.specified_files]
        else:
            csv_files = list(self.data_dir.glob("*_hdc.csv"))
            
        if not csv_files:
            print(f"警告: 在 {self.data_dir} 未找到 *_hdc.csv 文件")
            return

        damage_samples = []
        damage_labels = []
        damage_file_info = []
        
        normal_samples = []
        normal_labels = []
        normal_file_info = []
        
        total_windows = 0
        damage_windows = 0
        normal_windows = 0
        
        for csv_file in csv_files:
            # 1. 读取特征数据 (1_hdc.csv)
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"读取文件 {csv_file} 失败: {e}")
                continue

            base_name = csv_file.name.replace("_hdc.csv", "")
            
            # 2. 读取标签数据 (1_label.csv)
            label_file = self.data_dir / f"{base_name}_label.csv"
            if label_file.exists():
                try:
                    label_df = pd.read_csv(label_file)
                    # 确保标签列存在
                    if all(f'label_{i}' in label_df.columns for i in range(NUM_CHANNELS)):
                        # 按照 encoding 合并特征和标签
                        df = pd.merge(df, label_df, on='encoding', how='left').fillna(0)
                    else:
                        # 如果标签文件格式不对，填充 0
                        for i in range(NUM_CHANNELS):
                            df[f'label_{i}'] = 0
                except Exception as e:
                    print(f"读取或合并标签文件 {label_file} 失败: {e}")
                    # 回退到全 0 标签
                    for i in range(NUM_CHANNELS):
                        df[f'label_{i}'] = 0
            else:
                # 如果没有标签文件，所有通道标签设为 0（正常）
                for i in range(NUM_CHANNELS):
                    df[f'label_{i}'] = 0
            
            # 3. 确保所有必要的列都存在
            channel_cols = [f'channel_{i}' for i in range(NUM_CHANNELS)]
            label_cols = [f'label_{i}' for i in range(NUM_CHANNELS)]
            
            if not all(col in df.columns for col in channel_cols):
                print(f"文件 {csv_file.name} 缺少必要的通道列，已跳过")
                continue
            
            if not all(col in df.columns for col in label_cols):
                # 这种情况不应该发生，因为上面已经做了填充，但为了安全再次检查
                for col in label_cols:
                    if col not in df.columns:
                        df[col] = 0
            
            # 按照编码值排序，确保时序/空间连续性
            df = df.sort_values(by='encoding')

            channel_data = df[channel_cols].values.astype(np.float32)
            label_data = df[label_cols].values.astype(np.int64)
            encodings = df['encoding'].values

            # 4. 滑动窗口采样
            if len(df) < self.window_size:
                print(f"跳过文件 {csv_file.name}: 数据行数 ({len(df)}) 小于窗口大小 ({self.window_size})")
                continue

            # 步长为 1，确保每个点都能作为中心点被预测
            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                window_x = channel_data[start:end]
                
                # 获取中心行的 9 个通道标签 (index 63)
                center_labels = label_data[start + CENTER_IDX]
                center_encoding = encodings[start + CENTER_IDX]
                
                # 构建文件信息
                file_info = {
                    'file': str(csv_file),
                    'center_encoding': center_encoding,
                    'start': start
                }
                
                # 分离伤损样本和正常样本
                if any(center_labels > 0):
                    # 伤损样本
                    damage_windows += 1
                    damage_samples.append(window_x)
                    damage_labels.append(center_labels)
                    damage_file_info.append(file_info)
                else:
                    # 正常样本
                    normal_windows += 1
                    normal_samples.append(window_x)
                    normal_labels.append(center_labels)
                    normal_file_info.append(file_info)
                
                total_windows += 1
        
        # 确定需要的正常样本数量
        num_damage = len(damage_samples)
        num_normal = min(len(normal_samples), num_damage)
        
        # 随机选择正常样本
        import random
        if num_normal > 0:
            selected_indices = random.sample(range(len(normal_samples)), num_normal)
            selected_normal_samples = [normal_samples[i] for i in selected_indices]
            selected_normal_labels = [normal_labels[i] for i in selected_indices]
            selected_normal_file_info = [normal_file_info[i] for i in selected_indices]
        else:
            selected_normal_samples = []
            selected_normal_labels = []
            selected_normal_file_info = []
        
        # 合并样本
        self.samples = damage_samples + selected_normal_samples
        self.labels = damage_labels + selected_normal_labels
        self.file_info = damage_file_info + selected_normal_file_info
        
        # 打乱样本顺序
        if self.is_train:
            combined = list(zip(self.samples, self.labels, self.file_info))
            random.shuffle(combined)
            self.samples[:], self.labels[:], self.file_info[:] = zip(*combined)
        
        print(f"数据集加载完成: {len(damage_samples)} 个伤损样本 + {len(selected_normal_samples)} 个正常样本 = {len(self.samples)} 个总样本")
        print(f"总窗口数: {total_windows}, 伤损窗口: {damage_windows}, 正常窗口: {normal_windows}")

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

def augment_data_by_beam_offset():
    """
    数据增强：生成三份增强数据，分别是通道6、7、8各偏移20
    """
    print("\n[INFO] 正在进行数据增强 (修改BeamOffset)...")
    data_dirs = [TRAIN_DATA_DIR, TEST_DATA_DIR]
    
    # 定义增强配置：通道ID -> 偏移量
    augment_configs = [(6, 20), (7, 20), (8, 20)]
    
    for data_dir in data_dirs:
        original_files = get_json_files(data_dir)
        
        for original_file in original_files:
            # 跳过已增强的文件
            if '增强' in original_file.name:
                continue
            
            # 对每个配置生成增强数据
            for channel_id, offset in augment_configs:
                augmented_filename = f"{original_file.stem}增强_ch{channel_id}{original_file.suffix}"
                augmented_file = data_dir / augmented_filename
                
                # 检查增强文件是否已存在
                if augmented_file.exists():
                    continue
                
                # 读取原始JSON数据
                try:
                    with open(original_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"  ✗ 读取 {original_file.name} 失败: {e}")
                    continue
                
                # 处理BeamOffset
                beam_offset_key = 'BeamOffset'
                if isinstance(data, list) and len(data) > 0:
                    if beam_offset_key not in data[0]:
                        print(f"  [WARN] {original_file.name} 中没有找到 BeamOffset")
                        continue
                    bo = data[0][beam_offset_key]
                    is_list_data = True
                elif isinstance(data, dict):
                    if beam_offset_key not in data:
                        print(f"  [WARN] {original_file.name} 中没有找到 BeamOffset")
                        continue
                    bo = data[beam_offset_key]
                    is_list_data = False
                else:
                    print(f"  [WARN] {original_file.name} 数据格式不支持")
                    continue
                
                # 生成增强后的BeamOffset：只修改指定通道的偏移量
                augmented_bo = []
                for i in range(0, len(bo), 2):
                    current_channel = bo[i]
                    original_value = bo[i+1]
                    if current_channel == channel_id:
                        augmented_value = original_value + offset
                    else:
                        augmented_value = original_value
                    augmented_bo.extend([current_channel, augmented_value])
                
                # 更新数据中的BeamOffset
                if is_list_data:
                    data[0][beam_offset_key] = augmented_bo
                else:
                    data[beam_offset_key] = augmented_bo
                
                # 保存增强数据文件
                try:
                    with open(augmented_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"  ✓ 生成增强数据: {augmented_filename}")
                except Exception as e:
                    print(f"  ✗ 保存增强数据 {augmented_filename} 失败: {e}")
                    continue
                
                # 处理标注文件
                annotation_file = data_dir / f"{original_file.stem}标注{original_file.suffix}"
                if annotation_file.exists():
                    augmented_annotation_filename = f"{original_file.stem}增强_ch{channel_id}标注{original_file.suffix}"
                    augmented_annotation_file = data_dir / augmented_annotation_filename
                    
                    try:
                        shutil.copy(annotation_file, augmented_annotation_file)
                        print(f"  ✓ 生成增强标注: {augmented_annotation_filename}")
                    except Exception as e:
                        print(f"  ✗ 复制增强标注 {augmented_annotation_filename} 失败: {e}")

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

def clean_old_files(data_dirs, file_types="all"):
    """清理旧的中间文件，保持环境整洁"""
    # 根据 file_types 参数选择要删除的文件类型
    if file_types == "train":
        patterns = ["*_encoding.csv"]  # 不再删除hdc.csv和label.csv，训练和指标计算都需要用到
        print("\n[INFO] 正在清理旧的训练编码文件...")
    elif file_types == "predict":
        patterns = ["*_predicted_label.csv", "*_predicted.json"]
        print("\n[INFO] 正在清理旧的预测相关文件...")
    else: # all
        patterns = ["*_hdc.csv", "*_label.csv", "*_encoding.csv", "*_predicted_label.csv", "*_predicted.json"]
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
        'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'total_true': int(np.sum(true_mask)), 'total_pred': int(np.sum(pred_mask))
    }

def test_channel_offset(base_data_dir, test_file, channel_id, offset, device, target_encoding=None):
    """
    测试单个通道的偏移效果
    
    参数:
    - base_data_dir: 基础数据目录
    - test_file: 测试文件路径
    - channel_id: 通道ID (0-8)
    - offset: 偏移量
    - device: 设备
    - target_encoding: 目标点的编码值，用于提取相似度数据
    
    返回:
    - 包含precision, recall, f1的字典，以及目标点的相似度值
    """
    # 创建临时目录
    temp_dir = base_data_dir / f"temp_offset_{channel_id}_{offset}"
    temp_dir.mkdir(exist_ok=True)
    
    # 复制测试文件到临时目录
    temp_file = temp_dir / test_file.name
    shutil.copy(test_file, temp_file)
    
    # 读取并修改BeamOffset
    try:
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ 读取 {test_file.name} 失败: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    # 处理BeamOffset
    beam_offset_key = 'BeamOffset'
    if isinstance(data, list) and len(data) > 0:
        if beam_offset_key not in data[0]:
            print(f"  [WARN] {test_file.name} 中没有找到 BeamOffset")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
        bo = data[0][beam_offset_key]
        is_list_data = True
    elif isinstance(data, dict):
        if beam_offset_key not in data:
            print(f"  [WARN] {test_file.name} 中没有找到 BeamOffset")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
        bo = data[beam_offset_key]
        is_list_data = False
    else:
        print(f"  [WARN] {test_file.name} 数据格式不支持")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    # 修改特定通道的offset值
    modified_bo = []
    for i in range(0, len(bo), 2):
        current_channel = bo[i]
        current_value = bo[i+1]
        if current_channel == channel_id:
            modified_value = current_value + offset
        else:
            modified_value = current_value
        modified_bo.extend([current_channel, modified_value])
    
    # 更新数据中的BeamOffset
    if is_list_data:
        data[0][beam_offset_key] = modified_bo
    else:
        data[beam_offset_key] = modified_bo
    
    # 保存修改后的数据
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ✗ 保存修改后的数据失败: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    # 复制标注文件（如果存在）
    annotation_file = test_file.parent / f"{test_file.stem}标注{test_file.suffix}"
    if annotation_file.exists():
        temp_annotation = temp_dir / f"{test_file.stem}标注{test_file.suffix}"
        shutil.copy(annotation_file, temp_annotation)
    
    # 执行数据转换
    converter = UltrasonicDataConverter()
    annotation_path = str(temp_dir / f"{test_file.stem}标注{test_file.suffix}") if annotation_file.exists() else None
    try:
        converter.generate_all_csvs(str(temp_file), annotation_path)
    except Exception as e:
        print(f"  ✗ 数据转换失败: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    # 加载模型
    hdc_sys = HDCPointLevelModel(device=device)
    if MODEL_PATH.exists():
        try:
            hdc_sys.load(str(MODEL_PATH))
        except Exception as e:
            print(f"  ✗ 加载模型失败: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    # 执行预测
    hdc_csv = temp_dir / f"{test_file.stem}_hdc.csv"
    if not hdc_csv.exists():
        print(f"  ✗ 未找到转换后的HDC文件")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    ds = HDCPointLevelDataset(temp_dir, is_train=False, target_encoding=target_encoding)
    if len(ds) == 0:
        print(f"  ✗ 数据集为空")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    try:
        predictions, normal_sims, max_defect_sims, all_class_sims = hdc_sys.predict(ld, threshold=PREDICT_THRESHOLD, margin=PREDICT_MARGIN)
    except Exception as e:
        print(f"  ✗ 预测失败: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    # 提取目标点的相似度数据
    target_similarity = None
    if target_encoding is not None:
        for idx, info in enumerate(ds.file_info):
            if info['center_encoding'] == target_encoding:
                # 提取类别0、11、13的相似度
                class_0_sims = []
                class_11_sims = []
                class_13_sims = []
                
                for ch in range(len(all_class_sims[idx])):
                    # 获取当前通道的所有类别相似度
                    channel_sims = all_class_sims[idx][ch]
                    # 提取类别0的相似度
                    class_0_sims.append(channel_sims[0] if len(channel_sims) > 0 else 0.0)
                    # 提取类别11的相似度
                    class_11_sims.append(channel_sims[11] if len(channel_sims) > 11 else 0.0)
                    # 提取类别13的相似度
                    class_13_sims.append(channel_sims[13] if len(channel_sims) > 13 else 0.0)
                
                target_similarity = {
                    'normal': normal_sims[idx].tolist(),
                    'defect': max_defect_sims[idx].tolist(),
                    'class_0': class_0_sims,
                    'class_11': class_11_sims,
                    'class_13': class_13_sims
                }
                break
    
    # 计算指标
    label_file = temp_dir / f"{test_file.stem}_label.csv"
    if not label_file.exists():
        print(f"  ✗ 未找到标签文件")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': target_similarity}
    
    # 读取标签数据
    try:
        df_hdc = pd.read_csv(hdc_csv)
        df_label = pd.read_csv(label_file)
        df_merged = pd.merge(df_hdc[['encoding']], df_label, on='encoding', how='left').fillna(0)
        y_true_matrix = df_merged[[f'label_{i}' for i in range(NUM_CHANNELS)]].values
    except Exception as e:
        print(f"  ✗ 读取标签数据失败: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': target_similarity}
    
    # 准备预测矩阵
    y_pred_matrix = np.zeros((len(df_hdc), NUM_CHANNELS), dtype=np.int64)
    encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
    for idx, pred_row in enumerate(predictions):
        info = ds.file_info[idx]
        if info['center_encoding'] in encoding_to_idx:
            pred_idx = encoding_to_idx[info['center_encoding']]
            y_pred_matrix[pred_idx] = pred_row
    
    # 计算指标
    metrics = calculate_metrics(y_true_matrix, y_pred_matrix)
    metrics['similarity'] = target_similarity
    
    # 清理临时目录
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"  [WARN] 清理临时目录失败: {e}")
    
    return metrics

def run_channel_offset_test():
    """
    运行通道偏移测试，对通道6-8执行0-100偏移测试（步长4），记录指标并生成图表
    """
    print("\n" + "="*60)
    print(f"通道偏移测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")
    
    # 使用训练集数据
    test_files = get_json_files(TRAIN_DATA_DIR)
    if not test_files:
        print("[ERROR] 未找到训练文件")
        return
    
    # 选择第一个训练文件作为基础
    test_file = test_files[0]
    print(f"[INFO] 使用训练文件: {test_file.name}")
    
    # 创建结果目录
    offset_results_dir = RESULTS_DIR / "offset_tests"
    offset_results_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成偏移序列：0-100，步长4
    offsets = list(range(0, 101, 4))
    print(f"[INFO] 偏移范围: {offsets}")
    
    # 存储结果
    channel_results = {}
    for channel_id in range(6, 9):  # 只测试通道6-8
        channel_results[channel_id] = {'precisions': [], 'recalls': [], 'f1s': []}
        print(f"\n[INFO] 测试通道 {channel_id}...")
        
        for offset in tqdm(offsets, desc=f"通道 {channel_id}"):
            metrics = test_channel_offset(TRAIN_DATA_DIR, test_file, channel_id, offset, device)
            channel_results[channel_id]['precisions'].append(metrics['precision'])
            channel_results[channel_id]['recalls'].append(metrics['recall'])
            channel_results[channel_id]['f1s'].append(metrics['f1'])
    
    # 生成图表
    print("\n[INFO] 生成图表...")
    for channel_id in range(6, 9):  # 只生成通道6-8的图表
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 精准度
        ax1.plot(offsets, channel_results[channel_id]['precisions'], 'o-', color='blue')
        ax1.set_title('精准度 (Precision)')
        ax1.set_xlabel('偏移量 (Offset)')
        ax1.set_ylabel('精准度')
        ax1.grid(True)
        ax1.set_ylim(0, 1.1)
        
        # 召回率
        ax2.plot(offsets, channel_results[channel_id]['recalls'], 'o-', color='green')
        ax2.set_title('召回率 (Recall)')
        ax2.set_xlabel('偏移量 (Offset)')
        ax2.set_ylabel('召回率')
        ax2.grid(True)
        ax2.set_ylim(0, 1.1)
        
        # F1分数
        ax3.plot(offsets, channel_results[channel_id]['f1s'], 'o-', color='red')
        ax3.set_title('F1分数 (F1-Score)')
        ax3.set_xlabel('偏移量 (Offset)')
        ax3.set_ylabel('F1分数')
        ax3.grid(True)
        ax3.set_ylim(0, 1.1)
        
        # 整体设置
        fig.suptitle(f'通道 {channel_id} 偏移测试结果', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        chart_path = offset_results_dir / f"channel_{channel_id}_offset_test.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"  ✓ 已保存图表: {chart_path.name}")
    
    # 保存结果数据
    results_data = {}
    for channel_id in range(6, 9):  # 只保存通道6-8的结果
        results_data[channel_id] = {
            'precisions': channel_results[channel_id]['precisions'],
            'recalls': channel_results[channel_id]['recalls'],
            'f1s': channel_results[channel_id]['f1s']
        }
    
    results_file = offset_results_dir / "offset_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] 测试完成，结果已保存至: {offset_results_dir}")

def run_point_similarity_offset_test():
    """
    运行点级相似度偏移测试，对指定点执行不同偏移测试，记录相似度变化并生成图表
    """
    print("\n" + "="*60)
    print(f"点级相似度偏移测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 设置matplotlib中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")
    
    # 使用训练集数据
    test_files = get_json_files(TRAIN_DATA_DIR)
    if not test_files:
        print("[ERROR] 未找到训练文件")
        return
    
    # 选择包含标注的文件
    annotation_file_name = "170921_16229片段.xxc标注.json"
    target_data_file_name = "170921_16229片段.xxc.json"
    test_file = None
    
    # 首先尝试找到目标数据文件
    for file in test_files:
        if target_data_file_name in file.name:
            test_file = file
            break
    
    # 如果找不到目标文件，尝试找到包含相同前缀的文件
    if not test_file:
        for file in test_files:
            if "170921_16229片段" in file.name:
                test_file = file
                break
    
    if not test_file:
        # 如果找不到指定文件，使用第一个文件
        test_file = test_files[0]
    print(f"[INFO] 使用训练文件: {test_file.name}")
    
    # 创建结果目录
    similarity_results_dir = RESULTS_DIR / "similarity_offset_tests"
    print(f"[INFO] 结果目录: {similarity_results_dir}")
    similarity_results_dir.mkdir(exist_ok=True, parents=True)
    print(f"[INFO] 结果目录创建成功")
    
    # 生成偏移序列：0-100，步长4
    offsets = list(range(0, 40, 1))
    print(f"[INFO] 偏移范围: {offsets}")
    
    # 从标注文件中选择测试点
    test_points = [
        {'channel': 6, 'encoding': 592, 'depth': 90.0, 'name': '螺孔中心-通道6'},
        {'channel': 7, 'encoding': 853, 'depth': 79.0, 'name': '螺孔中心-通道7'},
        {'channel': 8, 'encoding': 875, 'depth': 64.0, 'name': '螺孔中心-通道8'}
    ]
    
    print(f"[INFO] 测试点数量: {len(test_points)}")
    for point in test_points:
        print(f"  - 通道 {point['channel']}: e={point['encoding']}, depth={point['depth']}, {point['name']}")
    
    # 存储结果
    point_results = {}
    for point in test_points:
        channel_id = point['channel']
        target_encoding = point['encoding']
        point_key = f"ch{channel_id}_e{target_encoding}"
        point_results[point_key] = {
            'channel': channel_id,
            'encoding': target_encoding,
            'name': point['name'],
            'normal_similarities': [],
            'defect_similarities': [],
            'class_0_similarities': [],
            'class_11_similarities': [],
            'class_13_similarities': []
        }
        
        print(f"\n[INFO] 测试点: {point['name']} (通道 {channel_id}, e={target_encoding})...")
        
        for offset in tqdm(offsets, desc=f"通道 {channel_id}, e={target_encoding}"):
            metrics = test_channel_offset(TRAIN_DATA_DIR, test_file, channel_id, offset, device, target_encoding)
            if metrics['similarity']:
                # 提取当前通道的相似度值
                normal_sim = metrics['similarity']['normal'][channel_id]
                defect_sim = metrics['similarity']['defect'][channel_id]
                class_0_sim = metrics['similarity']['class_0'][channel_id]
                class_11_sim = metrics['similarity']['class_11'][channel_id]
                class_13_sim = metrics['similarity']['class_13'][channel_id]
                
                point_results[point_key]['normal_similarities'].append(normal_sim)
                point_results[point_key]['defect_similarities'].append(defect_sim)
                point_results[point_key]['class_0_similarities'].append(class_0_sim)
                point_results[point_key]['class_11_similarities'].append(class_11_sim)
                point_results[point_key]['class_13_similarities'].append(class_13_sim)
            else:
                # 如果未找到目标点，添加默认值
                point_results[point_key]['normal_similarities'].append(0.0)
                point_results[point_key]['defect_similarities'].append(0.0)
                point_results[point_key]['class_0_similarities'].append(0.0)
                point_results[point_key]['class_11_similarities'].append(0.0)
                point_results[point_key]['class_13_similarities'].append(0.0)
    
    # 生成图表
    print("\n[INFO] 生成相似度变化图表...")
    for point_key, result in point_results.items():
        channel_id = result['channel']
        encoding = result['encoding']
        name = result['name']
        
        # 创建三列子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 类别0相似度
        ax1.plot(offsets, result['class_0_similarities'], 'o-', color='blue')
        ax1.set_title('类别0相似度', fontsize=14)
        ax1.set_xlabel('偏移量 (Offset)')
        ax1.set_ylabel('相似度')
        
        # 类别11相似度
        ax2.plot(offsets, result['class_11_similarities'], 'o-', color='green')
        ax2.set_title('类别11相似度', fontsize=14)
        ax2.set_xlabel('偏移量 (Offset)')
        ax2.set_ylabel('相似度')
        
        # 类别13相似度
        ax3.plot(offsets, result['class_13_similarities'], 'o-', color='purple')
        ax3.set_title('类别13相似度', fontsize=14)
        ax3.set_xlabel('偏移量 (Offset)')
        ax3.set_ylabel('相似度')
        
        # 计算每个子图的纵轴范围
        for ax, similarities in zip([ax1, ax2, ax3], [result['class_0_similarities'], result['class_11_similarities'], result['class_13_similarities']]):
            if similarities:
                min_sim = min(similarities)
                max_sim = max(similarities)
                # 添加一些边距，使图表更美观
                padding = (max_sim - min_sim) * 0.1
                if padding == 0:
                    padding = 0.1
                ax.set_ylim(min_sim - padding, max_sim + padding)
            else:
                ax.set_ylim(0, 1.1)
            ax.grid(True)
        
        # 整体标题
        fig.suptitle(f'{name} 相似度偏移测试结果', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        chart_path = similarity_results_dir / f"point_ch{channel_id}_e{encoding}_similarity_offset.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"  ✓ 已保存图表: {chart_path.name}")
    
    # 保存结果数据 - 转换numpy类型为Python原生类型
    import numpy as np
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 转换所有numpy类型
    converted_results = convert_numpy_types(point_results)
    
    results_file = similarity_results_dir / "point_similarity_offset_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    
    # 保存为CSV格式，以列的形式记录三个类别的相似度
    import pandas as pd
    for point_key, result in point_results.items():
        channel_id = result['channel']
        encoding = result['encoding']
        name = result['name']
        
        # 创建DataFrame
        df = pd.DataFrame({
            '偏移量 (Offset)': offsets,
            '类别0相似度': result['class_0_similarities'],
            '类别11相似度': result['class_11_similarities'],
            '类别13相似度': result['class_13_similarities']
        })
        
        # 保存为CSV文件
        csv_path = similarity_results_dir / f"point_ch{channel_id}_e{encoding}_similarity_offset.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"  ✓ 已保存CSV: {csv_path.name}")
    
    print(f"\n[INFO] 测试完成，结果已保存至: {similarity_results_dir}")

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
    learned_files = load_learned_files()
    new_trained_files = []
    
    print("\n[Step 1] 正在进行数据转换 (JSON -> CSV)...")
    # 执行数据增强
   #augment_data_by_beam_offset()
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
        # 如果需要训练，先清理旧的训练相关文件
        print(f"[INFO] 发现 {len(actual_new_files)} 个新文件，开始增量学习...")
        # 清理旧的训练相关文件
        clean_old_files([TRAIN_DATA_DIR], file_types="train")
        # 准备新文件的路径列表
        new_csv_files = [TRAIN_DATA_DIR / f"{Path(f).stem}_hdc.csv" for f in actual_new_files]
        # 根据配置选择使用相应的数据集类
        if TRAIN_STRATEGY == "damage_only":
            # 创建一个临时的 Dataset 只包含新文件，且只保留中心点为伤损的样本
            train_ds = HDCCenterDamageDataset(TRAIN_DATA_DIR, is_train=True, files=new_csv_files)
        elif TRAIN_STRATEGY == "mixed":
            # 创建一个临时的 Dataset 只包含新文件，包含伤损样本和同等数量的正常样本
            train_ds = HDCMixedDamageDataset(TRAIN_DATA_DIR, is_train=True, files=new_csv_files)
        else:
            # 创建一个临时的 Dataset 只包含新文件，使用所有样本
            train_ds = HDCPointLevelDataset(TRAIN_DATA_DIR, is_train=True, files=new_csv_files)
        if len(train_ds) == 0:
            print("[WARN] 过滤后训练集为空。")
        else:
            print(f"[INFO] 训练样本数: {len(train_ds)}")
            batch_size = 128 if torch.cuda.is_available() else 8
            train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            # 根据配置选择学习方式
            print(f"[INFO] 使用学习方式: {LEARNING_METHOD}")
            if LEARNING_METHOD == "ridge":
                hdc_sys.train(train_ld, method="ridge", alpha=RIDGE_ALPHA)
            else:
                # 使用 OnlineHD 学习方式
                hdc_sys.train(train_ld, epochs=EPOCHS, lr=LEARNING_RATE, method="onlinehd")
            
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
    # 清理旧的预测文件，只删除预测相关文件
    clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR], file_types="predict")
    
    # 定义处理流程：(目录, 是否计算指标)
    tasks = [
        (TRAIN_DATA_DIR, True, "训练集"),
        (TEST_DATA_DIR, True, "测试集")
    ]
    
    all_train_y_true = []
    all_train_y_pred = []

    # 设置预测 batch size
    eval_batch_size = 128 if torch.cuda.is_available() else 4
    print(f"[INFO] 预测 Batch Size: {eval_batch_size}")

    for data_dir, calc_metrics, label in tasks:
        print(f"\n--- 处理{label} ({data_dir}) ---")
        ds = HDCPointLevelDataset(data_dir, is_train=False) # 预测时不需要 shuffle
        if len(ds) == 0:
            print(f"[WARN] {label}为空，跳过。")
            continue
            
        ld = torch.utils.data.DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
        # 使用 Margin 逻辑预测：只有当"伤损类"相似度显著高于"正常类"时才保留
        # 基础阈值设低(0.1)，主要靠 Margin(0.5) 过滤误报
        predictions, normal_sims, max_defect_sims, all_class_sims = hdc_sys.predict(ld, threshold=PREDICT_THRESHOLD, margin=PREDICT_MARGIN) 
        
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
                'normal_sim': normal_sims[idx],
                'max_defect_sim': max_defect_sims[idx],
                'all_class_sims': all_class_sims[idx]  # 存储所有类别的相似度
            })

        for file_name, results in results_by_file.items():
            base_name = file_name.replace("_hdc.csv", "")
            hdc_file = data_dir / file_name
            label_file = data_dir / f"{base_name}_label.csv"
            
            # 读取特征
            df_hdc = pd.read_csv(hdc_file)
            
            # 准备预测矩阵
            y_pred_matrix = np.zeros((len(df_hdc), NUM_CHANNELS), dtype=np.int64)
            y_normal_sim_matrix = np.zeros((len(df_hdc), NUM_CHANNELS), dtype=np.float32)
            y_max_defect_sim_matrix = np.zeros((len(df_hdc), NUM_CHANNELS), dtype=np.float32)
            # 准备通道8三种类别的相似度矩阵
            y_class_0_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            y_class_11_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            y_class_13_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            
            # 填充预测值
            encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
            for res in results:
                if res['encoding'] in encoding_to_idx:
                    idx = encoding_to_idx[res['encoding']]
                    y_pred_matrix[idx] = res['pred']
                    y_normal_sim_matrix[idx] = res['normal_sim']
                    y_max_defect_sim_matrix[idx] = res['max_defect_sim']
                    
                    # 填充通道8三种类别的相似度
                    if 'all_class_sims' in res:
                        # 提取通道8的所有类别相似度
                        channel8_sims = res['all_class_sims'][8] if len(res['all_class_sims']) > 8 else []
                        if len(channel8_sims) > 0:
                            # 填充类别0、11、13的相似度
                            if len(channel8_sims) > 0:
                                y_class_0_sim_matrix[idx] = channel8_sims[0]  # 类别0
                            if len(channel8_sims) > 11:
                                y_class_11_sim_matrix[idx] = channel8_sims[11]  # 类别11
                            if len(channel8_sims) > 13:
                                y_class_13_sim_matrix[idx] = channel8_sims[13]  # 类别13
            
            # 保存预测结果 CSV
            pred_df = pd.DataFrame({'encoding': df_hdc['encoding']})
            for i in range(NUM_CHANNELS):
                pred_df[f'predicted_label_{i}'] = y_pred_matrix[:, i]
                # 合并为 "正常,伤损" 字符串
                sim_strs = [f"{n:.4f},{d:.4f}" for n, d in zip(y_normal_sim_matrix[:, i], y_max_defect_sim_matrix[:, i])]
                pred_df[f'similarity_{i}'] = sim_strs
            
            # 为通道8添加三种类别的相似度列
            pred_df['class_0_similarity'] = y_class_0_sim_matrix[:, 0]
            pred_df['class_11_similarity'] = y_class_11_sim_matrix[:, 0]
            pred_df['class_13_similarity'] = y_class_13_sim_matrix[:, 0]
            
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

def run_cross_validation(k_folds=5):
    """
    执行 K-Fold 交叉验证
    """
    print("="*60)
    print(f"HDC 交叉验证 (K={k_folds}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")

    # 1. 确保所有数据已转换为CSV
    print("\n[Prep] 检查数据文件...")
    converter = UltrasonicDataConverter()
    
    # 获取所有的训练数据文件 (作为总的池子)
    # 注意：这里我们只用 TRAIN_DATA_DIR 下的数据进行交叉验证
    all_json_files = get_json_files(TRAIN_DATA_DIR)
    
    valid_csv_files = []
    
    for json_file in tqdm(all_json_files, desc="数据准备"):
        hdc_csv = json_file.parent / f"{json_file.stem}_hdc.csv"
        
        # 如果对应的CSV不存在，尝试生成
        if not hdc_csv.exists():
            annotation_file = json_file.parent / f"{json_file.stem}标注{json_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            try:
                converter.generate_all_csvs(str(json_file), annotation_path)
            except Exception as e:
                print(f"  ✗ {json_file.name} 转换失败: {e}")
                continue
        
        if hdc_csv.exists():
            valid_csv_files.append(hdc_csv)

    if len(valid_csv_files) < k_folds:
        print(f"[ERROR] 有效文件数量 ({len(valid_csv_files)}) 小于 Fold 数量 ({k_folds})")
        return

    # 2. K-Fold 分割
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    # 转换为 numpy 数组方便索引
    files_array = np.array(valid_csv_files)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(files_array)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        
        train_files = files_array[train_idx]
        val_files = files_array[val_idx]
        
        print(f"训练集文件: {len(train_files)}, 验证集文件: {len(val_files)}")
        
        # 3. 准备数据集
        # 训练集：根据策略选择数据集类
        if TRAIN_STRATEGY == "damage_only":
            train_ds = HDCCenterDamageDataset(TRAIN_DATA_DIR, is_train=True, files=train_files)
        elif TRAIN_STRATEGY == "mixed":
            train_ds = HDCMixedDamageDataset(TRAIN_DATA_DIR, is_train=True, files=train_files)
        else:
            train_ds = HDCPointLevelDataset(TRAIN_DATA_DIR, is_train=True, files=train_files)
            
        # 验证集：总是使用完整数据集 (模拟真实测试)
        val_ds = HDCPointLevelDataset(TRAIN_DATA_DIR, is_train=False, files=val_files)
        
        if len(train_ds) == 0:
            print("[WARN] 训练集为空，跳过此 Fold")
            continue
        if len(val_ds) == 0:
            print("[WARN] 验证集为空，跳过此 Fold")
            continue
            
        # 4. 训练模型
        # 每个 Fold 重新初始化模型
        hdc_sys = HDCPointLevelModel(device=device)
        
        batch_size = 128 if torch.cuda.is_available() else 8
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        if LEARNING_METHOD == "ridge":
            hdc_sys.train(train_ld, method="ridge", alpha=RIDGE_ALPHA)
        else:
            hdc_sys.train(train_ld, epochs=EPOCHS, lr=LEARNING_RATE, method="onlinehd")
            
        # 5. 验证模型
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        predictions, _, _, _ = hdc_sys.predict(val_ld, threshold=PREDICT_THRESHOLD, margin=PREDICT_MARGIN)
        
        # 6. 计算指标
        # 优化：直接从 dataset 获取标签
        # val_ds.labels 是 [ (9,), (9,), ... ] 的列表
        # 需要将其堆叠为 [N, 9] 的 numpy 数组
        y_true_matrix = np.vstack(val_ds.labels)
        
        metrics = calculate_metrics(y_true_matrix, predictions)
        print(f"Fold {fold+1} Result: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'metrics': metrics
        })
        
    # 7. 汇总报告
    if not fold_results:
        print("[ERROR] 没有成功的 Fold")
        return

    print("\n" + "="*40)
    print("交叉验证汇总报告")
    print("="*40)
    
    precisions = [r['metrics']['precision'] for r in fold_results]
    recalls = [r['metrics']['recall'] for r in fold_results]
    f1s = [r['metrics']['f1'] for r in fold_results]
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_f1 = np.std(f1s)
    
    print(f"平均 Precision: {avg_precision:.4f} (+/- {std_precision:.4f})")
    print(f"平均 Recall:    {avg_recall:.4f} (+/- {std_recall:.4f})")
    print(f"平均 F1-Score:  {avg_f1:.4f} (+/- {std_f1:.4f})")
    print("-" * 40)
    for res in fold_results:
        m = res['metrics']
        print(f"Fold {res['fold']}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
    print("="*40)
    
    # 保存报告
    report_path = RESULTS_DIR / f"cross_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'avg_precision': float(avg_precision), 'std_precision': float(std_precision),
                'avg_recall': float(avg_recall), 'std_recall': float(std_recall),
                'avg_f1': float(avg_f1), 'std_f1': float(std_f1)
            },
            'folds': fold_results
        }, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 详细报告已保存至: {report_path}")

if __name__ == "__main__":
    run_full_test()
    # 直接运行点级相似度偏移测试
    #run_point_similarity_offset_test()
    #run_cross_validation(k_folds=5)
    # 运行通道偏移测试
    # run_channel_offset_test()
    # 直接运行点级相似度偏移测试
    # run_point_similarity_offset_test()
