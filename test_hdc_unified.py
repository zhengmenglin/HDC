#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 HDC 点级分类器测试脚本
支持单通道和多通道模式，通过配置文件控制
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
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

from hdc_point_level_classifier import (
    HDCPointLevelDataset, 
    HDCPointLevelModel, 
    WINDOW_SIZE, 
    NUM_TYPES, 
    CENTER_IDX,
    NUM_LEVELS,
    INPUT_MODE,
    DIMENSIONS,
)
# torchhd 由 hdc_point_level_classifier 统一导入（编码器/模型都在那里）
from ultrasonic_data_converter import UltrasonicDataConverter
from config import config
from hdc_utils import (
    set_seed, FLAW_ID_TO_NAME, load_learned_files, save_learned_files,
    clean_old_files, get_json_files, calculate_metrics, 
    convert_predicted_csv_to_json, load_csv_with_labels,
    get_channel_info, process_prediction_results, save_prediction_csv, generate_report
)

set_seed(42)

TRAIN_DATA_DIR = Path(config.get("data.train_dir"))
TEST_DATA_DIR = Path(config.get("data.test_dir"))
MODEL_PATH = Path(config.get("model.model_path"))
LEARNED_FILES_PATH = Path(config.get("model.learned_files_path"))
RESULTS_DIR = Path(config.get("test.results_dir"))

CHANNELS = config.get("model.channels", [0, 1, 2, 3, 4, 5, 6, 7, 8])
NUM_CHANNELS, IS_SINGLE_CHANNEL, CHANNEL_ID = get_channel_info(CHANNELS)

LEARNING_METHOD = config.get("model.learning_method", "onlinehd")
RIDGE_ALPHA = config.get("model.ridge_alpha", 1.0)
LEARNING_RATE = config.get("model.learning_rate", 0.1)
EPOCHS = config.get("model.epochs", 5)
REGEN_FREQ = config.get("model.regen_freq", 20)
REGEN_RATE = config.get("model.regen_rate", 0.04)

TRAIN_STRATEGY = config.get("test.train_strategy", "damage_only")
PREDICT_THRESHOLD = config.get("test.predict_threshold", 0)
PREDICT_MARGIN = config.get("test.predict_margin", 0)
# 新判据：训练集标定的 (md-ns) z-score 阈值；2.5 ≈ 正常波动之外 2.5σ（偏召回）
Z_THRESHOLD = config.get("test.z_threshold", 2.75)
NUM_LEVELS_CFG = config.get("model.num_levels", NUM_LEVELS)
INPUT_MODE_CFG = config.get("model.input_mode", INPUT_MODE)
DIMENSIONS_CFG = config.get("model.dimensions", DIMENSIONS)


class HDCUnifiedDataset(HDCPointLevelDataset):
    """
    统一数据集类，支持单通道和多通道模式
    """
    def __init__(self, data_dir, is_train=True, window_size=WINDOW_SIZE, step=1, 
                 files=None, target_encoding=None, train_strategy="damage_only",
                 channels=None):
        self.train_strategy = train_strategy
        self.channels = channels if channels is not None else CHANNELS
        self.num_channels = len(self.channels)
        self.is_single_channel = self.num_channels == 1
        super().__init__(data_dir, is_train, window_size, step, files, target_encoding,
                         channels=self.channels)
    
    def _load_data(self):
        if self.specified_files is not None:
            csv_files = [Path(f) for f in self.specified_files]
        else:
            csv_files = list(self.data_dir.glob("*_hdc.csv"))
        
        if not csv_files:
            print(f"警告: 在 {self.data_dir} 未找到 *_hdc.csv 文件")
            return

        all_normal_samples = []
        all_flaw_samples = []
        normal_file_infos = []
        flaw_file_infos = []

        for csv_file in csv_files:
            df, channel_cols, label_cols = self._load_csv_data(csv_file)
            if df is None:
                continue

            if len(df) < self.window_size:
                print(f"跳过文件 {csv_file.name}: 数据行数 ({len(df)}) 小于窗口大小 ({self.window_size})")
                continue

            channel_data = df[channel_cols].values.astype(np.float32)
            label_data = df[label_cols].values.astype(np.int64)
            encodings = df['encoding'].values

            normal_samples = []
            flaw_samples = []
            file_infos_n = []
            file_infos_f = []

            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                window_x = channel_data[start:end]
                center_encoding = encodings[start + CENTER_IDX]
                
                if self.target_encoding is not None and center_encoding != self.target_encoding:
                    continue
                
                file_info = {
                    'file': str(csv_file),
                    'center_encoding': center_encoding,
                    'start': start
                }
                
                if self.is_single_channel:
                    center_label = label_data[start + CENTER_IDX]
                    is_flaw = center_label > 0
                    sample_label = np.array([center_label.item() if hasattr(center_label, 'item') else center_label], dtype=np.int64)
                else:
                    center_labels = label_data[start + CENTER_IDX]
                    is_flaw = any(center_labels > 0)
                    sample_label = center_labels.astype(np.int64)
                
                if is_flaw:
                    flaw_samples.append((window_x, sample_label))
                    if not self.is_train:
                        file_infos_f.append(file_info)
                else:
                    normal_samples.append((window_x, sample_label))
                    if not self.is_train:
                        file_infos_n.append(file_info)

            if self.is_train:
                self._apply_train_strategy(csv_file.name, normal_samples, flaw_samples,
                                          all_normal_samples, all_flaw_samples)
            else:
                all_normal_samples.extend(normal_samples)
                all_flaw_samples.extend(flaw_samples)
                normal_file_infos.extend(file_infos_n)
                flaw_file_infos.extend(file_infos_f)

        self._finalize_samples(all_normal_samples, all_flaw_samples, 
                               normal_file_infos, flaw_file_infos)

    def _load_csv_data(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {e}")
            return None, None, None

        base_name = csv_file.name.replace("_hdc.csv", "")
        # 标签跟着 CSV 走；用 data_dir 会在“指定外部文件”时静默读不到标签
        label_file = csv_file.parent / f"{base_name}_label.csv"
        
        if self.is_single_channel:
            ch = self.channels[0]
            channel_cols = [f'channel_{ch}']
            label_cols = [f'label_{ch}']
        else:
            channel_cols = [f'channel_{ch}' for ch in self.channels]
            label_cols = [f'label_{ch}' for ch in self.channels]
        
        if label_file.exists():
            try:
                label_df = pd.read_csv(label_file)
                if self.is_single_channel:
                    ch = self.channels[0]
                    if f'label_{ch}' in label_df.columns:
                        df = pd.merge(df, label_df[["encoding", f'label_{ch}']], 
                                     on='encoding', how='left').fillna(0)
                    else:
                        df[f'label_{ch}'] = 0
                else:
                    merge_cols = ["encoding"] + [f'label_{ch}' for ch in self.channels 
                                                 if f'label_{ch}' in label_df.columns]
                    if len(merge_cols) > 1:
                        df = pd.merge(df, label_df[merge_cols], on='encoding', how='left').fillna(0)
                    for ch in self.channels:
                        if f'label_{ch}' not in df.columns:
                            df[f'label_{ch}'] = 0
            except Exception as e:
                print(f"读取或合并标签文件 {label_file} 失败: {e}")
                for col in label_cols:
                    df[col] = 0
        else:
            for col in label_cols:
                df[col] = 0
        
        if not all(col in df.columns for col in channel_cols):
            print(f"文件 {csv_file.name} 缺少必要的通道列，已跳过")
            return None, None, None
        
        for col in label_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df.sort_values(by='encoding')
        return df, channel_cols, label_cols

    def _apply_train_strategy(self, file_name, normal_samples, flaw_samples,
                              all_normal, all_flaw):
        if self.train_strategy == "damage_only":
            print(f"[INFO] damage_only: {file_name} 保留 {len(flaw_samples)} 个伤损样本")
            all_flaw.extend(flaw_samples)
        elif self.train_strategy == "mixed":
            flaw_count = len(flaw_samples)
            normal_count = len(normal_samples)
            if normal_count >= flaw_count:
                selected_normal = random.sample(normal_samples, flaw_count)
                print(f"[INFO] mixed: {file_name} 选取 {flaw_count} 伤损 + {flaw_count} 正常")
            else:
                selected_normal = normal_samples
                print(f"[INFO] mixed: {file_name} 选取 {flaw_count} 伤损 + {normal_count} 正常")
            all_flaw.extend(flaw_samples)
            all_normal.extend(selected_normal)
        else:
            total = len(normal_samples) + len(flaw_samples)
            print(f"[INFO] all: {file_name} 使用所有 {total} 个样本")
            all_normal.extend(normal_samples)
            all_flaw.extend(flaw_samples)

    def _finalize_samples(self, all_normal, all_flaw, normal_infos, flaw_infos):
        if self.is_train and self.train_strategy == "mixed":
            flaw_count = len(all_flaw)
            normal_count = len(all_normal)
            if normal_count > flaw_count:
                all_normal = random.sample(all_normal, flaw_count)
                print(f"[INFO] 最终平衡: {flaw_count} 伤损 + {flaw_count} 正常")
            else:
                print(f"[INFO] 最终平衡: {flaw_count} 伤损 + {normal_count} 正常")

        all_samples = all_normal + all_flaw
        
        if self.is_train:
            random.shuffle(all_samples)
        else:
            self.file_info.extend(normal_infos)
            self.file_info.extend(flaw_infos)
        
        for sample, label in all_samples:
            self.samples.append(sample)
            self.labels.append(label)
        
        channel_str = f"通道{self.channels[0]}" if self.is_single_channel else f"{self.num_channels}通道"
        print(f"数据集加载完成: {len(self.samples)} 个样本 ({channel_str})")
        
        if self.is_train:
            flaw_count = sum(1 for l in self.labels if self._is_flaw(l))
            normal_count = len(self.labels) - flaw_count
            print(f"[INFO] 样本分布: 正常 {normal_count} 个，伤损 {flaw_count} 个")

    def _is_flaw(self, label):
        if isinstance(label, np.ndarray):
            return any(label > 0)
        return label > 0


class HDCUnifiedModel(HDCPointLevelModel):
    """统一模型类：单通道/多通道都走父类的 Level 编码 + 类均值原型。

    旧版单通道用 OnlineHD(Sinusoid)、多通道用 Level 编码，是两套算法，
    指标不可比；现已统一。
    """
    def __init__(self, device='cpu', channels=None, num_levels=NUM_LEVELS,
                 input_mode=None, z_threshold=None, dimensions=None):
        ch = channels if channels is not None else CHANNELS
        super().__init__(
            device=device, channels=ch,
            dimensions=DIMENSIONS_CFG if dimensions is None else dimensions,
            num_levels=num_levels,
            input_mode=input_mode or INPUT_MODE,
            z_threshold=Z_THRESHOLD if z_threshold is None else z_threshold,
        )

    def train(self, train_loader, epochs=5, lr=0.1, method="centroid",
              alpha=1.0, regen_freq=20, regen_rate=0.04, reset=True):
        # 单通道/多通道统一走父类的“类均值 + 标定”。
        # 旧版单通道用 OnlineHD(Sinusoid)、多通道用 Level 编码，是两套算法，
        # 指标不可比；这里统一，regen_* 参数保留仅为兼容旧调用。
        super().train(train_loader, epochs=epochs, lr=lr, method=method,
                      alpha=alpha, reset=reset)

    def save(self, path):
        # 统一检查点格式：包含 encoder / channels / num_levels / class_counts / score_stats
        super().save(Path(path))

    def load(self, path, window_size=WINDOW_SIZE, regen_freq=20, regen_rate=0.04):
        # 统一加载：父类会校验 encoder 是否存在、维度/通道是否匹配。
        # 旧检查点若只有原型没有 encoder，会抛出明确错误，由调用方转为从零训练。
        super().load(path, window_size=window_size)


def run_unified_test(force_retrain=False, train_dir=None, test_dir=None,
                     model_path=None, results_dir=None, channels=None,
                     z_threshold=None):
    """训练 + 预测 + 评估。

    force_retrain=True：忽略已有模型与 learned_files，从零训练（修 bug/换编码后必须）。
    旧模型若缺少 encoder，会自动转为从零训练，不会只加载原型造成“编码与原型不同源”。
    """
    global TRAIN_DATA_DIR, TEST_DATA_DIR, MODEL_PATH, RESULTS_DIR
    global CHANNELS, NUM_CHANNELS, IS_SINGLE_CHANNEL, CHANNEL_ID, Z_THRESHOLD
    if channels:
        CHANNELS = list(channels)
        NUM_CHANNELS, IS_SINGLE_CHANNEL, CHANNEL_ID = get_channel_info(CHANNELS)
    if z_threshold is not None:
        Z_THRESHOLD = float(z_threshold)
    if train_dir is not None:
        TRAIN_DATA_DIR = Path(train_dir)
    if test_dir is not None:
        TEST_DATA_DIR = Path(test_dir)
    if model_path is not None:
        MODEL_PATH = Path(model_path)
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)

    channel_str = f"通道{CHANNEL_ID}" if IS_SINGLE_CHANNEL else f"{len(CHANNELS)}通道"
    print("="*60)
    print(f"HDC 点级分类器测试 ({channel_str}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")

    learned_files = load_learned_files(LEARNED_FILES_PATH)
    new_trained_files = []
    
    print("\n[Step 1] 正在进行数据转换 (JSON -> CSV)...")
    converter = UltrasonicDataConverter()
    for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
        json_files = get_json_files(data_dir)
        for json_file in json_files:
            file_name = json_file.name
            hdc_csv = json_file.parent / f"{json_file.stem}_hdc.csv"
            if data_dir == TRAIN_DATA_DIR and file_name in learned_files and hdc_csv.exists():
                continue
                
            annotation_file = json_file.parent / f"{json_file.stem}标注{json_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            try:
                converter.generate_all_csvs(str(json_file), annotation_path)
                if data_dir == TRAIN_DATA_DIR:
                    new_trained_files.append(file_name)
            except Exception as e:
                print(f"  ✗ {json_file.name} 转换失败: {e}")

    print("\n[Step 2] 正在进行 HDC 模型训练...")
    hdc_sys = HDCUnifiedModel(device=device, channels=CHANNELS,
                              num_levels=NUM_LEVELS_CFG,
                              input_mode=INPUT_MODE_CFG,
                              z_threshold=Z_THRESHOLD,
                              dimensions=DIMENSIONS_CFG)

    model_loaded = False
    if MODEL_PATH.exists() and not force_retrain:
        print(f"[INFO] 尝试加载现有模型: {MODEL_PATH}")
        try:
            hdc_sys.load(str(MODEL_PATH), regen_freq=REGEN_FREQ, regen_rate=REGEN_RATE)
            model_loaded = True
        except Exception as e:
            print(f"[WARN] 旧模型不可用（{e}）→ 改为从零训练")
            force_retrain = True

    if force_retrain:
        # 换编码/修 bug 后必须重训；清空学习记录，避免“以为学过”而跳过
        learned_files = set()
        save_learned_files(learned_files, LEARNED_FILES_PATH)

    actual_new_files = (get_json_files(TRAIN_DATA_DIR)
                        if (force_retrain or not model_loaded)
                        else [f for f in get_json_files(TRAIN_DATA_DIR)
                              if f.name not in learned_files])

    if not actual_new_files:
        if not model_loaded:
            raise RuntimeError("没有可训练数据，也没有可用模型；请检查 --train-dir")
        print("[INFO] 没有新数据需要训练，直接使用已加载模型评估。")
    else:
        print(f"[INFO] 待训练文件 {len(actual_new_files)} 个"
              f"（{'从零训练' if not model_loaded else '增量累加'}）...")
        clean_old_files([TRAIN_DATA_DIR], file_types="train")
        new_csv_files = [Path(f).parent / f"{Path(f).stem}_hdc.csv" for f in actual_new_files]
        
        train_ds = HDCUnifiedDataset(
            TRAIN_DATA_DIR, 
            is_train=True, 
            files=new_csv_files,
            train_strategy=TRAIN_STRATEGY,
            channels=CHANNELS
        )
        
        if len(train_ds) == 0:
            print("[WARN] 过滤后训练集为空。")
        else:
            print(f"[INFO] 训练样本数: {len(train_ds)}")
            batch_size = 128 if torch.cuda.is_available() else 8
            train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            print(f"[INFO] 使用学习方式: {LEARNING_METHOD}")
            hdc_sys.train(
                train_ld, 
                epochs=EPOCHS, 
                lr=LEARNING_RATE, 
                method=LEARNING_METHOD,
                alpha=RIDGE_ALPHA,
                regen_freq=REGEN_FREQ,
                regen_rate=REGEN_RATE,
                reset=not model_loaded,
            )
            
            hdc_sys.save(str(MODEL_PATH))
            
            for f in actual_new_files:
                learned_files.add(f.name)
            save_learned_files(learned_files, LEARNED_FILES_PATH)
            print(f"[INFO] 已更新学习记录: {len(learned_files)} 个文件")

    if hdc_sys.encoder is None:
        raise RuntimeError("训练未完成：编码器未初始化（训练样本为空？），无法进入预测")

    print("\n[Step 3] 正在对数据集进行点级预测与转换...")
    clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR], file_types="predict")
    
    tasks = [
        (TRAIN_DATA_DIR, "训练集"),
        (TEST_DATA_DIR, "测试集"),
    ]
    # 旧版把 train+test 的矩阵拼在一起算一个总指标，报告却写“训练集”，
    # 分母不可追溯；这里按 split 分开统计，测试集单独出报告。
    split_true = {"训练集": [], "测试集": []}
    split_pred = {"训练集": [], "测试集": []}

    eval_batch_size = 128 if torch.cuda.is_available() else 4
    print(f"[INFO] 预测 Batch Size: {eval_batch_size}")

    for data_dir, label in tasks:
        print(f"\n--- 处理{label} ({data_dir}) ---")
        ds = HDCUnifiedDataset(data_dir, is_train=False, channels=CHANNELS)
        if len(ds) == 0:
            print(f"[WARN] {label}为空，跳过。")
            continue
            
        ld = torch.utils.data.DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
        predictions, normal_sims, max_defect_sims, all_class_sims = hdc_sys.predict(ld)
        
        results_by_file = {}
        for idx, pred_row in enumerate(predictions):# 遍历每个样本的预测结果
            info = ds.file_info[idx]
            file_name = Path(info['file']).name
            if file_name not in results_by_file:
                results_by_file[file_name] = []
            results_by_file[file_name].append({
                'encoding': info['center_encoding'],
                'pred_idx': idx
            })

        for file_name, results in results_by_file.items():# 遍历每个文件的预测结果
            base_name = file_name.replace("_hdc.csv", "")
            hdc_file = data_dir / file_name
            label_file = data_dir / f"{base_name}_label.csv"
            
            df_hdc = pd.read_csv(hdc_file)
            
            y_pred_matrix, y_normal_sim_matrix, y_max_defect_sim_matrix, y_class_sims_dict = \
                process_prediction_results(# 处理预测结果，生成预测矩阵和相似度矩阵
                    predictions, normal_sims, max_defect_sims, all_class_sims,
                    ds.file_info, df_hdc, CHANNELS, file_name=file_name
                )
            
            pred_csv_path = data_dir / f"{base_name}_predicted_label.csv"# 预测结果的CSV文件路径
            save_prediction_csv(# 保存预测结果到CSV文件
                df_hdc, y_pred_matrix, y_normal_sim_matrix, y_max_defect_sim_matrix,
                y_class_sims_dict, CHANNELS, pred_csv_path
            )
            
            original_json = None
            for ext in ['.json', '.xxc.json']:# 遍历可能的JSON文件扩展名
                potential_json = data_dir / f"{base_name}{ext}"
                if potential_json.exists():
                    original_json = potential_json
                    break
            
            if original_json:
                output_json = data_dir / f"{base_name}_predicted.json"
                convert_predicted_csv_to_json(pred_csv_path, original_json, output_json, hdc_file, # 转换预测结果CSV为JSON文件
                                             channel_ids=CHANNELS)

            if label_file.exists():
                df_label = pd.read_csv(label_file)
                label_cols = [f'label_{ch}' for ch in CHANNELS]
                
                if all(col in df_label.columns for col in label_cols):
                    df_merged = pd.merge(df_hdc[['encoding']], df_label[['encoding'] + label_cols], 
                                        on='encoding', how='left').fillna(0)
                    y_true_matrix = df_merged[label_cols].values
                    
                    metrics = calculate_metrics(y_true_matrix, y_pred_matrix)
                    split_true[label].append(y_true_matrix)
                    split_pred[label].append(y_pred_matrix)
                    
                    print(f"  文件: {file_name} -> P: {metrics['precision']:.4f}, "
                          f"R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, "
                          f"true={metrics['total_true']}, pred={metrics['total_pred']}")

    for label in ("训练集", "测试集"):
        if not split_true[label]:
            continue
        final_y_true = np.vstack(split_true[label])
        final_y_pred = np.vstack(split_pred[label])
        total_metrics = calculate_metrics(final_y_true, final_y_pred)
        print(f"\n[{label}] P={total_metrics['precision']:.4f} "
              f"R={total_metrics['recall']:.4f} F1={total_metrics['f1']:.4f} "
              f"true={total_metrics['total_true']} pred={total_metrics['total_pred']} "
              f"tp={total_metrics['tp']}")
        if label == "测试集":
            generate_report(total_metrics, MODEL_PATH, RESULTS_DIR, CHANNELS,
                            split_name="测试集")
    if not split_true["测试集"] and split_true["训练集"]:
        # 没有测试标签时，至少把训练集结果落一份，避免“无报告”
        total_metrics = calculate_metrics(np.vstack(split_true["训练集"]),
                                          np.vstack(split_pred["训练集"]))
        generate_report(total_metrics, MODEL_PATH, RESULTS_DIR, CHANNELS,
                        split_name="训练集")


def augment_data_by_beam_offset():
    """数据增强：生成三份增强数据，分别是通道6、7、8各偏移20"""
    print("\n[INFO] 正在进行数据增强 (修改BeamOffset)...")
    data_dirs = [TRAIN_DATA_DIR, TEST_DATA_DIR]
    
    augment_configs = [(6, 20), (7, 20), (8, 20)]
    
    for data_dir in data_dirs:
        original_files = get_json_files(data_dir)
        
        for original_file in original_files:
            if '增强' in original_file.name:
                continue
            
            for channel_id, offset in augment_configs:
                augmented_filename = f"{original_file.stem}增强_ch{channel_id}{original_file.suffix}"
                augmented_file = data_dir / augmented_filename
                
                if augmented_file.exists():
                    continue
                
                try:
                    with open(original_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"  ✗ 读取 {original_file.name} 失败: {e}")
                    continue
                
                beam_offset_key = 'BeamOffset'
                if isinstance(data, list) and len(data) > 0:
                    if beam_offset_key not in data[0]:
                        continue
                    bo = data[0][beam_offset_key]
                    is_list_data = True
                elif isinstance(data, dict):
                    if beam_offset_key not in data:
                        continue
                    bo = data[beam_offset_key]
                    is_list_data = False
                else:
                    continue
                
                augmented_bo = []
                for i in range(0, len(bo), 2):
                    current_channel = bo[i]
                    original_value = bo[i+1]
                    if current_channel == channel_id:
                        augmented_value = original_value + offset
                    else:
                        augmented_value = original_value
                    augmented_bo.extend([current_channel, augmented_value])
                
                if is_list_data:
                    data[0][beam_offset_key] = augmented_bo
                else:
                    data[beam_offset_key] = augmented_bo
                
                try:
                    with open(augmented_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"  ✓ 生成增强数据: {augmented_filename}")
                except Exception as e:
                    print(f"  ✗ 保存增强数据 {augmented_filename} 失败: {e}")
                
                annotation_file = data_dir / f"{original_file.stem}标注{original_file.suffix}"
                if annotation_file.exists():
                    augmented_annotation_filename = f"{original_file.stem}增强_ch{channel_id}标注{original_file.suffix}"
                    augmented_annotation_file = data_dir / augmented_annotation_filename
                    
                    try:
                        import shutil as sh
                        sh.copy(annotation_file, augmented_annotation_file)
                        print(f"  ✓ 生成增强标注: {augmented_annotation_filename}")
                    except Exception as e:
                        print(f"  ✗ 复制增强标注失败: {e}")


def test_channel_offset(base_data_dir, test_file, channel_id, offset, device, target_encoding=None):
    """测试单个通道的偏移效果"""
    import shutil as sh
    import matplotlib.pyplot as plt
    
    temp_dir = base_data_dir / f"temp_offset_{channel_id}_{offset}"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / test_file.name
    sh.copy(test_file, temp_file)
    
    try:
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ 读取 {test_file.name} 失败: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    beam_offset_key = 'BeamOffset'
    if isinstance(data, list) and len(data) > 0:
        if beam_offset_key not in data[0]:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
        bo = data[0][beam_offset_key]
        is_list_data = True
    elif isinstance(data, dict):
        if beam_offset_key not in data:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
        bo = data[beam_offset_key]
        is_list_data = False
    else:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    modified_bo = []
    for i in range(0, len(bo), 2):
        current_channel = bo[i]
        current_value = bo[i+1]
        if current_channel == channel_id:
            modified_value = current_value + offset
        else:
            modified_value = current_value
        modified_bo.extend([current_channel, modified_value])
    
    if is_list_data:
        data[0][beam_offset_key] = modified_bo
    else:
        data[beam_offset_key] = modified_bo
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    annotation_file = test_file.parent / f"{test_file.stem}标注{test_file.suffix}"
    if annotation_file.exists():
        temp_annotation = temp_dir / f"{test_file.stem}标注{test_file.suffix}"
        sh.copy(annotation_file, temp_annotation)
    
    converter = UltrasonicDataConverter()
    annotation_path = str(temp_dir / f"{test_file.stem}标注{test_file.suffix}") if annotation_file.exists() else None
    try:
        converter.generate_all_csvs(str(temp_file), annotation_path)
    except Exception as e:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    hdc_sys = HDCUnifiedModel(device=device, channels=CHANNELS)
    if MODEL_PATH.exists():
        try:
            hdc_sys.load(str(MODEL_PATH), regen_freq=REGEN_FREQ, regen_rate=REGEN_RATE)
        except Exception as e:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    hdc_csv = temp_dir / f"{test_file.stem}_hdc.csv"
    if not hdc_csv.exists():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    ds = HDCUnifiedDataset(temp_dir, is_train=False, target_encoding=target_encoding, channels=CHANNELS)
    if len(ds) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    ld = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    try:
        predictions, normal_sims, max_defect_sims, all_class_sims = hdc_sys.predict(ld, threshold=PREDICT_THRESHOLD, margin=PREDICT_MARGIN)
    except Exception as e:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': None}
    
    target_similarity = None
    if target_encoding is not None:
        for idx, info in enumerate(ds.file_info):
            if info['center_encoding'] == target_encoding:
                class_0_sims = []
                class_11_sims = []
                class_13_sims = []
                
                for ch in range(len(all_class_sims[idx])):
                    channel_sims = all_class_sims[idx][ch]
                    class_0_sims.append(channel_sims[0] if len(channel_sims) > 0 else 0.0)
                    class_11_sims.append(channel_sims[11] if len(channel_sims) > 11 else 0.0)
                    class_13_sims.append(channel_sims[13] if len(channel_sims) > 13 else 0.0)
                
                target_similarity = {
                    'normal': normal_sims[idx].tolist(),
                    'defect': max_defect_sims[idx].tolist(),
                    'class_0': class_0_sims,
                    'class_11': class_11_sims,
                    'class_13': class_13_sims
                }
                break
    
    label_file = temp_dir / f"{test_file.stem}_label.csv"
    if not label_file.exists():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': target_similarity}
    
    try:
        df_hdc = pd.read_csv(hdc_csv)
        df_label = pd.read_csv(label_file)
        label_cols = [f'label_{ch}' for ch in CHANNELS]
        df_merged = pd.merge(df_hdc[['encoding']], df_label[['encoding'] + label_cols], on='encoding', how='left').fillna(0)
        y_true_matrix = df_merged[label_cols].values
    except Exception as e:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'similarity': target_similarity}
    
    y_pred_matrix = np.zeros((len(df_hdc), len(CHANNELS)), dtype=np.int64)
    encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
    for idx, pred_row in enumerate(predictions):
        info = ds.file_info[idx]
        if info['center_encoding'] in encoding_to_idx:
            pred_idx = encoding_to_idx[info['center_encoding']]
            y_pred_matrix[pred_idx] = pred_row
    
    metrics = calculate_metrics(y_true_matrix, y_pred_matrix)
    metrics['similarity'] = target_similarity
    
    try:
        import shutil as sh
        sh.rmtree(temp_dir)
    except Exception:
        pass
    
    return metrics


def run_channel_offset_test():
    """运行通道偏移测试，对通道6-8执行0-100偏移测试（步长4）"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    print("\n" + "="*60)
    print(f"通道偏移测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")
    
    test_files = get_json_files(TRAIN_DATA_DIR)
    if not test_files:
        print("[ERROR] 未找到训练文件")
        return
    
    test_file = test_files[0]
    print(f"[INFO] 使用训练文件: {test_file.name}")
    
    offset_results_dir = RESULTS_DIR / "offset_tests"
    offset_results_dir.mkdir(exist_ok=True, parents=True)
    
    offsets = list(range(0, 101, 4))
    print(f"[INFO] 偏移范围: {offsets}")
    
    channel_results = {}
    for channel_id in range(6, 9):
        channel_results[channel_id] = {'precisions': [], 'recalls': [], 'f1s': []}
        print(f"\n[INFO] 测试通道 {channel_id}...")
        
        for offset in tqdm(offsets, desc=f"通道 {channel_id}"):
            metrics = test_channel_offset(TRAIN_DATA_DIR, test_file, channel_id, offset, device)
            channel_results[channel_id]['precisions'].append(metrics['precision'])
            channel_results[channel_id]['recalls'].append(metrics['recall'])
            channel_results[channel_id]['f1s'].append(metrics['f1'])
    
    print("\n[INFO] 生成图表...")
    for channel_id in range(6, 9):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.plot(offsets, channel_results[channel_id]['precisions'], 'o-', color='blue')
        ax1.set_title('精准度 (Precision)')
        ax1.set_xlabel('偏移量 (Offset)')
        ax1.set_ylabel('精准度')
        ax1.grid(True)
        ax1.set_ylim(0, 1.1)
        
        ax2.plot(offsets, channel_results[channel_id]['recalls'], 'o-', color='green')
        ax2.set_title('召回率 (Recall)')
        ax2.set_xlabel('偏移量 (Offset)')
        ax2.set_ylabel('召回率')
        ax2.grid(True)
        ax2.set_ylim(0, 1.1)
        
        ax3.plot(offsets, channel_results[channel_id]['f1s'], 'o-', color='red')
        ax3.set_title('F1分数 (F1-Score)')
        ax3.set_xlabel('偏移量 (Offset)')
        ax3.set_ylabel('F1分数')
        ax3.grid(True)
        ax3.set_ylim(0, 1.1)
        
        fig.suptitle(f'通道 {channel_id} 偏移测试结果', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        chart_path = offset_results_dir / f"channel_{channel_id}_offset_test.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"  ✓ 已保存图表: {chart_path.name}")
    
    results_data = {}
    for channel_id in range(6, 9):
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
    """运行点级相似度偏移测试"""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    
    print("\n" + "="*60)
    print(f"点级相似度偏移测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")
    
    test_files = get_json_files(TRAIN_DATA_DIR)
    if not test_files:
        print("[ERROR] 未找到训练文件")
        return
    
    target_data_file_name = "170921_16229片段.xxc.json"
    test_file = None
    
    for file in test_files:
        if target_data_file_name in file.name:
            test_file = file
            break
    
    if not test_file:
        for file in test_files:
            if "170921_16229片段" in file.name:
                test_file = file
                break
    
    if not test_file:
        test_file = test_files[0]
    print(f"[INFO] 使用训练文件: {test_file.name}")
    
    similarity_results_dir = RESULTS_DIR / "similarity_offset_tests"
    similarity_results_dir.mkdir(exist_ok=True, parents=True)
    
    offsets = list(range(0, 40, 1))
    print(f"[INFO] 偏移范围: {offsets}")
    
    test_points = [
        {'channel': 6, 'encoding': 592, 'depth': 90.0, 'name': '螺孔中心-通道6'},
        {'channel': 7, 'encoding': 853, 'depth': 79.0, 'name': '螺孔中心-通道7'},
        {'channel': 8, 'encoding': 875, 'depth': 64.0, 'name': '螺孔中心-通道8'}
    ]
    
    print(f"[INFO] 测试点数量: {len(test_points)}")
    for point in test_points:
        print(f"  - 通道 {point['channel']}: e={point['encoding']}, depth={point['depth']}, {point['name']}")
    
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
                point_results[point_key]['normal_similarities'].append(0.0)
                point_results[point_key]['defect_similarities'].append(0.0)
                point_results[point_key]['class_0_similarities'].append(0.0)
                point_results[point_key]['class_11_similarities'].append(0.0)
                point_results[point_key]['class_13_similarities'].append(0.0)
    
    print("\n[INFO] 生成相似度变化图表...")
    for point_key, result in point_results.items():
        channel_id = result['channel']
        encoding = result['encoding']
        name = result['name']
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.plot(offsets, result['class_0_similarities'], 'o-', color='blue')
        ax1.set_title('类别0相似度', fontsize=14)
        ax1.set_xlabel('偏移量 (Offset)')
        ax1.set_ylabel('相似度')
        
        ax2.plot(offsets, result['class_11_similarities'], 'o-', color='green')
        ax2.set_title('类别11相似度', fontsize=14)
        ax2.set_xlabel('偏移量 (Offset)')
        ax2.set_ylabel('相似度')
        
        ax3.plot(offsets, result['class_13_similarities'], 'o-', color='purple')
        ax3.set_title('类别13相似度', fontsize=14)
        ax3.set_xlabel('偏移量 (Offset)')
        ax3.set_ylabel('相似度')
        
        for ax, similarities in zip([ax1, ax2, ax3], [result['class_0_similarities'], result['class_11_similarities'], result['class_13_similarities']]):
            if similarities:
                min_sim = min(similarities)
                max_sim = max(similarities)
                padding = (max_sim - min_sim) * 0.1
                if padding == 0:
                    padding = 0.1
                ax.set_ylim(min_sim - padding, max_sim + padding)
            else:
                ax.set_ylim(0, 1.1)
            ax.grid(True)
        
        fig.suptitle(f'{name} 相似度偏移测试结果', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        chart_path = similarity_results_dir / f"point_ch{channel_id}_e{encoding}_similarity_offset.png"
        plt.savefig(chart_path)
        plt.close()
        print(f"  ✓ 已保存图表: {chart_path.name}")
    
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
    
    converted_results = convert_numpy_types(point_results)
    
    results_file = similarity_results_dir / "point_similarity_offset_test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, ensure_ascii=False, indent=2)
    
    for point_key, result in point_results.items():
        channel_id = result['channel']
        encoding = result['encoding']
        name = result['name']
        
        df = pd.DataFrame({
            '偏移量 (Offset)': offsets,
            '类别0相似度': result['class_0_similarities'],
            '类别11相似度': result['class_11_similarities'],
            '类别13相似度': result['class_13_similarities']
        })
        
        csv_path = similarity_results_dir / f"point_ch{channel_id}_e{encoding}_similarity_offset.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"  ✓ 已保存CSV: {csv_path.name}")
    
    print(f"\n[INFO] 测试完成，结果已保存至: {similarity_results_dir}")


def run_cross_validation(k_folds=5):
    """执行 K-Fold 交叉验证"""
    from sklearn.model_selection import KFold
    
    print("="*60)
    print(f"HDC 交叉验证 (K={k_folds}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")

    print("\n[Prep] 检查数据文件...")
    converter = UltrasonicDataConverter()
    
    all_json_files = get_json_files(TRAIN_DATA_DIR)
    
    valid_csv_files = []
    
    for json_file in tqdm(all_json_files, desc="数据准备"):
        hdc_csv = json_file.parent / f"{json_file.stem}_hdc.csv"
        
        if not hdc_csv.exists():
            annotation_file = json_file.parent / f"{json_file.stem}标注{json_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            try:
                converter.generate_all_csvs(str(json_file), annotation_path)
            except Exception as e:
                continue
        
        if hdc_csv.exists():
            valid_csv_files.append(hdc_csv)

    if len(valid_csv_files) < k_folds:
        print(f"[ERROR] 有效文件数量 ({len(valid_csv_files)}) 小于 Fold 数量 ({k_folds})")
        return

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    files_array = np.array(valid_csv_files)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(files_array)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        
        train_files = files_array[train_idx]
        val_files = files_array[val_idx]
        
        print(f"训练集文件: {len(train_files)}, 验证集文件: {len(val_files)}")
        
        train_ds = HDCUnifiedDataset(
            TRAIN_DATA_DIR, is_train=True, files=train_files,
            train_strategy=TRAIN_STRATEGY, channels=CHANNELS
        )
        val_ds = HDCUnifiedDataset(
            TRAIN_DATA_DIR, is_train=False, files=val_files,
            channels=CHANNELS
        )
        
        if len(train_ds) == 0:
            print("[WARN] 训练集为空，跳过此 Fold")
            continue
        if len(val_ds) == 0:
            print("[WARN] 验证集为空，跳过此 Fold")
            continue
            
        hdc_sys = HDCUnifiedModel(device=device, channels=CHANNELS)
        
        batch_size = 128 if torch.cuda.is_available() else 8
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        hdc_sys.train(
            train_ld, epochs=EPOCHS, lr=LEARNING_RATE, method=LEARNING_METHOD,
            alpha=RIDGE_ALPHA, regen_freq=REGEN_FREQ, regen_rate=REGEN_RATE
        )
            
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        predictions, _, _, _ = hdc_sys.predict(val_ld, threshold=PREDICT_THRESHOLD, margin=PREDICT_MARGIN)
        
        y_true_matrix = np.vstack(val_ds.labels)
        
        metrics = calculate_metrics(y_true_matrix, predictions)
        print(f"Fold {fold+1} Result: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'metrics': metrics
        })
        
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
    import argparse
    ap = argparse.ArgumentParser(description="HDC 点级分类训练/评估")
    ap.add_argument("--train-dir", help="训练数据目录（默认取 config）")
    ap.add_argument("--test-dir", help="测试数据目录（默认取 config）")
    ap.add_argument("--model", help="模型路径（默认取 config）")
    ap.add_argument("--results-dir", help="报告输出目录（默认取 config）")
    ap.add_argument("--channels", help="逗号分隔的 0-based 通道号，如 6,7,8（默认取 config）")
    ap.add_argument("--z-threshold", type=float,
                    help="伤损判据的 z 阈值（默认取 config，2.5σ）")
    ap.add_argument("--force-retrain", action="store_true",
                    help="忽略已有模型/学习记录，从零训练（换编码或修 bug 后必用）")
    ap.add_argument("--no-plots", action="store_true", help="不生成相似度图表")
    args = ap.parse_args()

    chs = [int(x) for x in args.channels.split(",")] if args.channels else None
    run_unified_test(
        force_retrain=args.force_retrain,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_path=args.model,
        results_dir=args.results_dir,
        channels=chs,
        z_threshold=args.z_threshold,
    )
    if not args.no_plots:
        try:
            from plot_similarity_charts import main as plot_similarity_charts_main
            plot_similarity_charts_main()
        except Exception as e:
            print(f"[WARN] 相似度图表生成失败（不影响主流程）: {e}")
