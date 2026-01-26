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
    数据增强：对原始数据的BeamOffset值添加随机整数(0-10)，生成增强数据文件
    """
    print("\n[INFO] 正在进行数据增强 (修改BeamOffset)...")
    data_dirs = [TRAIN_DATA_DIR, TEST_DATA_DIR]
    
    for data_dir in data_dirs:
        original_files = get_json_files(data_dir)
        
        for original_file in original_files:
            # 跳过已增强的文件
            if '增强' in original_file.name:
                continue
            
            augmented_filename = f"{original_file.stem}增强{original_file.suffix}"
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
            
            # 生成增强后的BeamOffset：每个通道值加上0-10的随机整数
            augmented_bo = []
            for i in range(0, len(bo), 2):
                channel_id = bo[i]
                original_value = bo[i+1]
                random_increment = random.randint(0, 10)
                augmented_value = original_value + random_increment
                augmented_bo.extend([channel_id, augmented_value])
            
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
                augmented_annotation_filename = f"{original_file.stem}增强标注{original_file.suffix}"
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
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'total_true': np.sum(true_mask), 'total_pred': np.sum(pred_mask)
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
    learned_files = load_learned_files()
    new_trained_files = []
    
    print("\n[Step 1] 正在进行数据转换 (JSON -> CSV)...")
    # 执行数据增强
    augment_data_by_beam_offset()
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
        # 创建一个临时的 Dataset 只包含新文件，且只保留中心点为伤损的样本
        #train_ds = HDCCenterDamageDataset(TRAIN_DATA_DIR, is_train=True, files=new_csv_files)
                # 创建一个临时的 Dataset 只包含新文件
        train_ds = HDCPointLevelDataset(TRAIN_DATA_DIR, is_train=True, files=new_csv_files)
        if len(train_ds) == 0:
            print("[WARN] 过滤后训练集为空。")
        else:
            print(f"[INFO] 训练样本数: {len(train_ds)}")
            batch_size = 128 if torch.cuda.is_available() else 8
            train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            # 减少训练轮数用于测试 OnlineHD 学习方式
            hdc_sys.train(train_ld, epochs=5, lr=0.1)
            
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
        (TEST_DATA_DIR, False, "测试集")
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
        # 使用 Margin 逻辑预测：只有当“伤损类”相似度显著高于“正常类”时才保留
        # 基础阈值设低(0.1)，主要靠 Margin(0.5) 过滤误报
        predictions, normal_sims, max_defect_sims = hdc_sys.predict(ld, threshold=0, margin=0) 
        
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
                'max_defect_sim': max_defect_sims[idx]
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
            
            # 填充预测值
            encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
            for res in results:
                if res['encoding'] in encoding_to_idx:
                    idx = encoding_to_idx[res['encoding']]
                    y_pred_matrix[idx] = res['pred']
                    y_normal_sim_matrix[idx] = res['normal_sim']
                    y_max_defect_sim_matrix[idx] = res['max_defect_sim']
            
            # 保存预测结果 CSV
            pred_df = pd.DataFrame({'encoding': df_hdc['encoding']})
            for i in range(NUM_CHANNELS):
                pred_df[f'predicted_label_{i}'] = y_pred_matrix[:, i]
                # 合并为 "正常,伤损" 字符串
                sim_strs = [f"{n:.4f},{d:.4f}" for n, d in zip(y_normal_sim_matrix[:, i], y_max_defect_sim_matrix[:, i])]
                pred_df[f'similarity_{i}'] = sim_strs
            
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
