#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只使用通道8进行训练和预测的测试脚本
包含：
1. 数据清理与初始化
2. JSON 到 CSV 数据转换 (利用 UltrasonicDataConverter)
3. HDC 模型预测 (只使用通道8)
4. 性能指标计算 (Precision, Recall, F1-score)
5. 记录类别0、11、13的相似度
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import random
import shutil
import json

# 添加 torchhd 库路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

# 导入 HDC 模型相关类
from hdc_point_level_classifier import (
    HDCPointLevelDataset, 
    HDCPointLevelModel, 
    WINDOW_SIZE, 
    NUM_CHANNELS, 
    NUM_TYPES, 
    CENTER_IDX
)

# 导入 torchhd 相关模块
import torchhd
from torchhd import functional as F
from torchhd.models import Centroid
from torchhd.classifiers import NeuralHD
from torchhd.embeddings import Random, Level

# 导入 PyTorch 标准库的 normalize 函数
import torch.nn.functional as F_pt

# 导入数据转换器
from ultrasonic_data_converter import UltrasonicDataConverter

# 导入配置管理
from config import Config

# ======================================
# 设置随机种子以确保结果可复现
# ======================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42)

# 加载配置
config = Config()

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
LEARNING_METHOD = config.get("model.learning_method", "neuralhd")
RIDGE_ALPHA = config.get("model.ridge_alpha", 1.0)
LEARNING_RATE = config.get("model.learning_rate", 0.37)
EPOCHS = config.get("model.epochs", 120)
# NeuralHD 相关参数
REGEN_FREQ = config.get("model.regen_freq", 20)
REGEN_RATE = config.get("model.regen_rate", 0.04)

# 测试配置
TRAIN_STRATEGY = config.get("test.train_strategy", "damage_only")
PREDICT_THRESHOLD = config.get("test.predict_threshold", 0)
PREDICT_MARGIN = config.get("test.predict_margin", 0)

# ======================================
# 数据集类（只使用通道8）
# ======================================
class HDCPointLevelDatasetChannel8(HDCPointLevelDataset):
    """
    只使用通道8的点级分类 HDC 数据集类
    """
    def __init__(self, data_dir, is_train=True, window_size=WINDOW_SIZE, step=1, files=None, target_encoding=None, train_strategy="damage_only"):
        # 先设置 train_strategy，再调用父类初始化
        self.train_strategy = train_strategy
        super().__init__(data_dir, is_train, window_size, step, files, target_encoding)
    
    def _load_data(self):
        """
        加载 CSV 数据并生成滑动窗口样本，只使用通道8的数据。
        """
        if self.specified_files is not None:
            csv_files = [Path(f) for f in self.specified_files]
        else:
            csv_files = list(self.data_dir.glob("*_hdc.csv"))
        
        if not csv_files:
            print(f"警告: 在 {self.data_dir} 未找到 *_hdc.csv 文件")
            return

        # 初始化全局样本存储
        all_normal_samples = []
        all_flaw_samples = []

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
                    if 'label_8' in label_df.columns:
                        # 按照 encoding 合并特征和标签
                        df = pd.merge(df, label_df[["encoding", "label_8"]], on='encoding', how='left').fillna(0)
                    else:
                        # 如果标签文件格式不对，填充 0
                        df['label_8'] = 0
                except Exception as e:
                    print(f"读取或合并标签文件 {label_file} 失败: {e}")
                    # 回退到全 0 标签
                    df['label_8'] = 0
            else:
                # 如果没有标签文件，通道8标签设为 0（正常）
                df['label_8'] = 0
            
            # 3. 确保所有必要的列都存在
            channel_col = 'channel_8'
            label_col = 'label_8'
            
            if channel_col not in df.columns:
                print(f"文件 {csv_file.name} 缺少通道8列，已跳过")
                continue
            
            if label_col not in df.columns:
                # 这种情况不应该发生，因为上面已经做了填充，但为了安全再次检查
                df[label_col] = 0
            
            # 按照编码值排序，确保时序/空间连续性
            df = df.sort_values(by='encoding')

            # 只使用通道8的数据
            channel_data = df[[channel_col]].values.astype(np.float32)
            label_data = df[[label_col]].values.astype(np.int64)
            encodings = df['encoding'].values

            # 4. 滑动窗口采样
            if len(df) < self.window_size:
                print(f"跳过文件 {csv_file.name}: 数据行数 ({len(df)}) 小于窗口大小 ({self.window_size})")
                continue

            # 为当前文件创建临时样本存储
            normal_samples = []
            flaw_samples = []
            normal_file_infos = []
            flaw_file_infos = []

            # 步长为 1，确保每个点都能作为中心点被预测
            for start in range(0, len(df) - self.window_size + 1, self.step):
                end = start + self.window_size
                window_x = channel_data[start:end]
                
                # 获取中心行的通道8标签 (index 63)
                center_label = label_data[start + CENTER_IDX]
                center_encoding = encodings[start + CENTER_IDX]
                
                # 如果指定了目标编码，只保留包含目标编码的窗口
                if self.target_encoding is not None and center_encoding != self.target_encoding:
                    continue
                
                # 准备文件信息
                file_info = {
                    'file': str(csv_file),
                    'center_encoding': center_encoding,
                    'start': start
                }
                
                # 根据中心点标签分类存储样本
                if center_label <= 0:
                    # 正常样本
                    normal_samples.append((window_x, center_label))
                    if not self.is_train:
                        normal_file_infos.append(file_info)
                else:
                    # 伤损样本
                    flaw_samples.append((window_x, center_label))
                    if not self.is_train:
                        flaw_file_infos.append(file_info)

            # 根据训练策略处理样本
            if self.is_train:
                if self.train_strategy == "damage_only":
                    # 只保留伤损样本
                    print(f"[INFO] 应用 damage_only 策略: 从文件 {csv_file.name} 中只保留 {len(flaw_samples)} 个伤损样本")
                    all_flaw_samples.extend(flaw_samples)
                elif self.train_strategy == "mixed":
                    # 平衡样本：选取与伤损样本数量相同的正常样本
                    flaw_count = len(flaw_samples)
                    normal_count = len(normal_samples)
                    if normal_count >= flaw_count:
                        # 随机选取相同数量的正常样本
                        selected_normal = random.sample(normal_samples, flaw_count)
                        print(f"[INFO] 应用 mixed 策略: 从文件 {csv_file.name} 中选取 {flaw_count} 个伤损样本和 {flaw_count} 个正常样本")
                    else:
                        # 如果正常样本不足，使用所有正常样本
                        selected_normal = normal_samples
                        print(f"[INFO] 应用 mixed 策略: 从文件 {csv_file.name} 中选取 {flaw_count} 个伤损样本和所有 {normal_count} 个正常样本")
                    all_flaw_samples.extend(flaw_samples)
                    all_normal_samples.extend(selected_normal)
                elif self.train_strategy == "all":
                    # 使用所有样本
                    print(f"[INFO] 应用 all 策略: 从文件 {csv_file.name} 中使用所有 {len(normal_samples) + len(flaw_samples)} 个样本")
                    all_normal_samples.extend(normal_samples)
                    all_flaw_samples.extend(flaw_samples)
            else:
                # 预测时使用所有样本
                all_normal_samples.extend(normal_samples)
                all_flaw_samples.extend(flaw_samples)
                # 预测时添加文件信息
                self.file_info.extend(normal_file_infos)
                self.file_info.extend(flaw_file_infos)

        # 合并并处理所有样本
        if self.is_train and self.train_strategy == "mixed":
            # 确保混合策略下样本平衡
            flaw_count = len(all_flaw_samples)
            normal_count = len(all_normal_samples)
            if normal_count > flaw_count:
                # 再次平衡，确保正常样本数量等于伤损样本数量
                all_normal_samples = random.sample(all_normal_samples, flaw_count)
                print(f"[INFO] 最终平衡: 混合策略使用 {flaw_count} 个伤损样本和 {flaw_count} 个正常样本")
            else:
                print(f"[INFO] 最终平衡: 混合策略使用 {flaw_count} 个伤损样本和 {normal_count} 个正常样本")

        # 合并所有样本
        all_samples = all_normal_samples + all_flaw_samples
        
        # 只在训练模式下打乱样本顺序
        if self.is_train:
            random.shuffle(all_samples)

        # 将样本添加到全局列表
        for sample, label in all_samples:
            self.samples.append(sample)
            self.labels.append(label)
        
        print(f"数据集加载完成: {len(self.samples)} 个样本 (只使用通道8)")
        if self.is_train:
            # 统计最终样本分布
            flaw_count = sum(1 for label in self.labels if label > 0)
            normal_count = len(self.labels) - flaw_count
            print(f"[INFO] 最终样本分布: 正常样本 {normal_count} 个，伤损样本 {flaw_count} 个")

# ======================================
# 模型类（只使用通道8）
# ======================================
class HDCPointLevelModelChannel8(HDCPointLevelModel):
    """
    只使用通道8的点级 HDC 模型
    """
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        # 使用 NeuralHD 分类器（只通道8）
        self.model = None
        self.device = device
    
    def train_with_ridge(self, train_loader, alpha=1.0):
        """
        使用岭回归学习方式训练模型（只使用通道8）
        """
        print(f"开始训练点级 HDC 模型 (只使用通道8, 岭回归学习方式, alpha={alpha})...")
        # 初始化编码器
        try:
            first_batch = next(iter(train_loader))
            sample_shape = first_batch[0].shape[1:] # [window_size, 1] (只通道8)
            if self.encoder is None:
                self.encoder = torchhd.embeddings.Random(1, self.dimensions)
                # 重新初始化编码器，只使用通道8
                self.encoder = self._create_channel8_encoder(sample_shape[0])
                self.encoder = self.encoder.to(self.device)
        except StopIteration:
            print("训练集为空，无法训练")
            return

        # 收集所有训练数据
        all_hv = []
        all_targets = []
        
        self.encoder.eval()
        with torch.no_grad():
            with tqdm(train_loader, desc="收集训练数据") as pbar:
                for samples, targets in pbar:
                    samples = samples.to(self.device)
                    targets = targets.to(self.device) # [batch, 1]
                    
                    hv = self.encoder(samples) # [batch, dim]
                    all_hv.append(hv)
                    all_targets.append(targets)
        
        # 合并所有数据
        all_hv = torch.cat(all_hv, dim=0)
        all_targets = torch.cat(all_targets, dim=0).squeeze(1) # [batch]
        
        # 对通道8执行岭回归学习
        ch = 0 # 只有一个通道（通道8）
        ch_targets = all_targets
        
        # 转换为one-hot编码
        one_hot = torch.zeros(len(ch_targets), NUM_TYPES, device=self.device)
        one_hot[torch.arange(len(ch_targets)), ch_targets] = 1
        
        # 使用岭回归计算权重
        weights = torchhd.functional.ridge_regression(all_hv, one_hot, alpha=alpha)
        
        # 更新模型权重
        self.models[ch].weight.copy_(weights)
        
        # 进行归一化，以便后续推理进行相似度计算
        self.models[ch].normalize()
            
        print("岭回归训练完成。")
    
    def train(self, train_loader, epochs=120, lr=0.37, method="neuralhd", alpha=1.0, regen_freq=20, regen_rate=0.04):
        """
        训练模型（只使用通道8）
        """
        # 从配置文件获取分类器类型和学习方式
        config_classifier = config.get("model.classifier", "OnlineHD")
        config_learning_method = config.get("model.learning_method", "onlinehd")
        
        print(f"[INFO] 配置文件设置: 分类器={config_classifier}, 学习方式={config_learning_method}")
        
        if method == "ridge":
            self.train_with_ridge(train_loader, alpha=alpha)
        else:
            try:
                first_batch = next(iter(train_loader))
                sample_shape = first_batch[0].shape[1:] # [window_size, 1] (只通道8)
                input_dim = sample_shape[0] * sample_shape[1] # window_size * 1
                
                if self.model is None:
                    # 根据配置选择分类器
                    if config_classifier == "OnlineHD" and config_learning_method == "onlinehd":
                        # 使用 OnlineHD 学习方式
                        print(f"开始训练点级 HDC 模型 (只使用通道8, Epochs={epochs}, OnlineHD 学习方式)...")
                        from torchhd.classifiers import OnlineHD
                        self.model = OnlineHD(
                            n_features=input_dim,
                            n_dimensions=self.dimensions,
                            n_classes=NUM_TYPES,
                            epochs=epochs,
                            lr=lr,
                            device=self.device
                        )
                    else:
                        # 默认使用 NeuralHD 学习方式
                        print(f"开始训练点级 HDC 模型 (只使用通道8, Epochs={epochs}, NeuralHD 学习方式)...")
                        from torchhd.classifiers import NeuralHD
                        self.model = NeuralHD(
                            n_features=input_dim,
                            n_dimensions=self.dimensions,
                            n_classes=NUM_TYPES,
                            regen_freq=regen_freq,
                            regen_rate=regen_rate,
                            epochs=epochs,
                            lr=lr,
                            device=self.device
                        )
            except StopIteration:
                print("训练集为空，无法训练")
                return

            # 准备数据加载器适配分类器的 fit 方法
            class AdaptedDataLoader:
                def __init__(self, data_loader):
                    self.data_loader = data_loader
                
                def __iter__(self):
                    for samples, targets in self.data_loader:
                        # 调整输入形状：[batch, window_size, 1] -> [batch, window_size*1]
                        adapted_samples = samples.view(samples.size(0), -1)
                        # 调整目标形状：[batch, 1] -> [batch]
                        adapted_targets = targets.squeeze(1)
                        yield adapted_samples, adapted_targets

            # 适配数据加载器
            adapted_loader = AdaptedDataLoader(train_loader)
            
            # 使用分类器的 fit 方法训练
            self.model.fit(adapted_loader)
            
            print("训练完成。")
    
    def predict(self, test_loader, threshold=0.1, margin=0.0):
        """
        预测类别（只使用通道8）
        返回: (predictions, normal_sims, max_defect_sims, all_class_sims)
        """
        print(f"开始预测 (只使用通道8, Threshold={threshold})...")
        if self.model is None:
            print("错误: 模型未初始化")
            return [], [], [], []
            
        self.model.eval()
        
        all_preds = []
        all_normal_sims = []
        all_max_defect_sims = []
        all_class_sims = []  # 存储所有类别的相似度
        
        with torch.no_grad():
            for samples, _ in tqdm(test_loader, desc="预测中"):
                samples = samples.to(self.device)
                # 调整输入形状：[batch, window_size, 1] -> [batch, window_size*1]
                adapted_samples = samples.view(samples.size(0), -1)
                
                # 使用 NeuralHD 的 forward 方法获取 logits
                logits = self.model(adapted_samples)
                
                # 计算余弦相似度（NeuralHD 的 forward 方法返回的是点积，需要归一化）
                # 获取模型权重并归一化
                normalized_weights = F_pt.normalize(self.model.model.weight, dim=1)
                # 计算输入的编码并归一化
                encoded = self.model.encoder(adapted_samples)
                normalized_encoded = F_pt.normalize(encoded, dim=1)
                # 计算余弦相似度
                output = torch.matmul(normalized_encoded, normalized_weights.t())
                
                # 获取正常类(0)的相似度
                normal_sim = output[:, 0]
                
                # 获取除正常类外最相似的伤损类
                defect_output = output[:, 1:] # 排除索引0
                max_defect_sim, defect_pred = torch.max(defect_output, dim=1)
                defect_pred = defect_pred + 1 # 还原索引
                
                # 判定逻辑：
                # 伤损类相似度必须大于基础阈值
                is_defect = (max_defect_sim > threshold)
                
                final_pred = torch.zeros_like(normal_sim, dtype=torch.long)
                final_pred[is_defect] = defect_pred[is_defect]
                
                # 存储结果
                all_preds.append(final_pred.unsqueeze(1).cpu())
                all_normal_sims.append(normal_sim.unsqueeze(1).cpu())
                all_max_defect_sims.append(max_defect_sim.unsqueeze(1).cpu())
                all_class_sims.append(output.unsqueeze(1).cpu())  # 存储所有类别的相似度
        
        return (
            torch.cat(all_preds, dim=0).numpy(), 
            torch.cat(all_normal_sims, dim=0).numpy(), 
            torch.cat(all_max_defect_sims, dim=0).numpy(),
            torch.cat(all_class_sims, dim=0).numpy()  # 返回所有类别的相似度
        )
    
    def _create_channel8_encoder(self, window_size):
        """
        创建只使用通道8的编码器
        """
        from torchhd.embeddings import Random, Level
        from torchhd import functional as F
        
        class Channel8Encoder(torch.nn.Module):
            def __init__(self, dimensions, window_size):
                super().__init__()
                self.dimensions = dimensions
                self.window_size = window_size
                # 数值超向量 (256个级别)
                self.value_hvs = Level(256, dimensions, high=256)
                # 位置超向量 (用于绑定)
                self.position_hvs = Random(window_size, dimensions)
            
            def forward(self, x):
                # x: [batch, window_size, 1]
                batch_size = x.size(0)
                
                # 1. 数值编码: [batch, window_size, 1, dimensions]
                val_hvs = self.value_hvs(x.long()) 
                
                # 2. 聚合通道信息 (只有通道8)
                # val_hvs: [batch, window_size, 1, dimensions] -> [batch, window_size, dimensions]
                time_step_hvs = val_hvs.squeeze(2)
                
                # 3. 绑定位置信息: [batch, window_size, dimensions]
                # position_hvs: [window_size, dimensions] -> [1, window_size, dimensions]
                pos_hvs = self.position_hvs.weight.unsqueeze(0)
                final_bound = F.bind(time_step_hvs, pos_hvs)
                
                # 4. 聚合整个窗口: [batch, dimensions]
                window_hv = torch.sum(final_bound, dim=1)
                
                return F.normalize(window_hv)
        
        return Channel8Encoder(self.dimensions, window_size)
    
    def load(self, path, window_size=WINDOW_SIZE):
        """
        加载模型（只使用通道8）
        """
        # 使用 weights_only=False 以兼容自定义类 (如 torchhd 的 Tensor 类型)
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # 兼容旧版本 PyTorch 不支持 weights_only 参数
            checkpoint = torch.load(path, map_location=self.device)
            
        self.dimensions = checkpoint['dimensions']
        
        # 初始化 NeuralHD 模型
        input_dim = window_size * 1  # window_size * 1 (只通道8)
        self.model = NeuralHD(
            n_features=input_dim,
            n_dimensions=self.dimensions,
            n_classes=NUM_TYPES,
            device=self.device
        )
        
        # 加载模型状态
        if 'model' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model'])
                print(f"模型已从 {path} 加载 (只使用通道8)。")
            except Exception as e:
                print(f"[WARN] 加载模型状态失败: {e}")
        else:
            print(f"[WARN] 模型文件中未找到模型状态，使用新初始化的 NeuralHD 模型。")

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
    y_true, y_pred: [N, 1] 形状的矩阵
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
            # 建立 (encoding) -> depth 的映射（只使用通道8）
            for _, row in df_hdc.iterrows():
                enc = int(row['encoding'])
                depth_map[enc] = row['channel_8']
        except Exception as e:
            print(f"  [WARN] 读取 {hdc_csv_path} 失败: {e}")

    # 3. 按标签分组收集点
    # 格式: { label_id: [ {e, channel, depth, amplitude}, ... ] }
    annotations_groups = {}
    
    for _, row in df_pred.iterrows():
        encoding = int(row['encoding'])
        label_col = 'predicted_label_8'
        if label_col in row and row[label_col] > 0:
            label_id = int(row[label_col])
            offset = beam_offset_map.get(8, 0) # 只使用通道8的偏移
            original_e = int(encoding + offset)
            
            depth = depth_map.get(encoding, 0.0)
            
            if label_id not in annotations_groups:
                annotations_groups[label_id] = []
            
            annotations_groups[label_id].append({
                "e": original_e,
                "channel": 8,
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
        chs = [8] # 只使用通道8
        
        annotations_list.append({
            "id": str(annotation_id),
            "type": "伤损" if label_id < 12 else "结构",
            "subtype": {"name": name, "id": str(label_id)},
            "points": points,
            "start_E": min(es),
            "end_E": max(es),
            "channels": chs,
            "timestamp": pd.Timestamp.now().isoformat(),
            "source": "predicted"
        })
        annotation_id += 1
    
    result = {"annotations": annotations_list}
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # print(f"  ✓ {os.path.basename(output_json_path)}")

# ======================================
# 主测试函数
# ======================================
def test_channel8_only():
    print("="*60)
    print(f"HDC 点级分类器测试 (只使用通道8) - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] 使用设备: {device}")

    # 1. 环境准备与数据转换
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
    print("\n[Step 2] 正在进行 HDC 模型训练 (只使用通道8)...")
    hdc_sys = HDCPointLevelModelChannel8(device=device)
    
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
        # 创建训练数据集
        train_ds = HDCPointLevelDatasetChannel8(
            TRAIN_DATA_DIR, 
            is_train=True, 
            files=new_csv_files,
            train_strategy=TRAIN_STRATEGY
        )
        
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
                # 使用 NeuralHD 学习方式
                hdc_sys.train(
                    train_ld, 
                    epochs=EPOCHS, 
                    lr=LEARNING_RATE, 
                    method="neuralhd",
                    regen_freq=REGEN_FREQ,
                    regen_rate=REGEN_RATE
                )
            
            # 保存模型
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            # 保存模型状态
            model_state = {
                'dimensions': hdc_sys.dimensions,
                'model': hdc_sys.model.state_dict() if hdc_sys.model else None
            }
            torch.save(model_state, str(MODEL_PATH))
            print(f"[INFO] 模型已保存至: {MODEL_PATH}")
            
            # 更新已学习列表
            for f in actual_new_files:
                learned_files.add(f.name)
            save_learned_files(learned_files)
            print(f"[INFO] 已更新学习记录: {len(learned_files)} 个文件")

    # 3. 执行预测与转换
    print("\n[Step 3] 正在对数据集进行点级预测与转换 (只使用通道8)...")
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
        ds = HDCPointLevelDatasetChannel8(data_dir, is_train=False) # 预测时不需要 shuffle
        if len(ds) == 0:
            print(f"[WARN] {label}为空，跳过。")
            continue
            
        ld = torch.utils.data.DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
        # 使用 Margin 逻辑预测：只有当"伤损类"相似度显著高于"正常类"时才保留
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
            y_pred_matrix = np.zeros((len(df_hdc), 1), dtype=np.int64)
            y_normal_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            y_max_defect_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            # 准备类别0、11、13的相似度矩阵
            y_class_0_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            y_class_11_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            y_class_13_sim_matrix = np.zeros((len(df_hdc), 1), dtype=np.float32)
            
            # 填充预测值和相似度
            encoding_to_idx = {enc: i for i, enc in enumerate(df_hdc['encoding'].values)}
            for res in results:
                if res['encoding'] in encoding_to_idx:
                    idx = encoding_to_idx[res['encoding']]
                    y_pred_matrix[idx] = res['pred']
                    y_normal_sim_matrix[idx] = res['normal_sim']
                    y_max_defect_sim_matrix[idx] = res['max_defect_sim']
                    
                    # 提取类别0、11、13的相似度
                    all_sims = res['all_class_sims'][0]  # [num_classes]
                    if len(all_sims) > 0:
                        y_class_0_sim_matrix[idx] = all_sims[0]  # 类别0
                    if len(all_sims) > 11:
                        y_class_11_sim_matrix[idx] = all_sims[11]  # 类别11
                    if len(all_sims) > 13:
                        y_class_13_sim_matrix[idx] = all_sims[13]  # 类别13
            
            # 保存预测结果 CSV
            pred_df = pd.DataFrame({'encoding': df_hdc['encoding']})
            # 添加通道8的预测结果
            pred_df['predicted_label_8'] = y_pred_matrix[:, 0]
            # 添加相似度列
            pred_df['similarity_8'] = [f"{n:.4f},{d:.4f}" for n, d in zip(y_normal_sim_matrix[:, 0], y_max_defect_sim_matrix[:, 0])]
            # 添加类别0、11、13的相似度列
            pred_df['class_0_similarity'] = y_class_0_sim_matrix[:, 0]
            pred_df['class_11_similarity'] = y_class_11_sim_matrix[:, 0]
            pred_df['class_13_similarity'] = y_class_13_sim_matrix[:, 0]
            
            pred_csv_path = data_dir / f"{base_name}_predicted_label.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            
            # 特别输出：针对 170921_16229片段.xxc 寻找 class_13_similarity 最高且大于 0.9 的 encoding
            if "170921_16229片段.xxc" in base_name:
                max_sim_13 = pred_df['class_13_similarity'].max()
                if max_sim_13 > 0.9:
                    max_sim_idx = pred_df['class_13_similarity'].idxmax()
                    target_encoding = pred_df.loc[max_sim_idx, 'encoding']
                    print(f"\n" + "!"*30)
                    print(f"[FOUND] 文件: {base_name}")
                    print(f"[FOUND] class_13_similarity 最大值: {max_sim_13:.4f}")
                    print(f"[FOUND] 对应的 encoding: {target_encoding}")
                    print("!"*30 + "\n")
            
            # --- 转换为 JSON ---
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

            # --- 计算指标 (仅限训练集) ---
            if calc_metrics and label_file.exists():
                try:
                    df_label = pd.read_csv(label_file)
                    df_merged = pd.merge(df_hdc[['encoding']], df_label[['encoding', 'label_8']], on='encoding', how='left').fillna(0)
                    y_true_matrix = df_merged[['label_8']].values
                    
                    metrics = calculate_metrics(y_true_matrix, y_pred_matrix)
                    all_train_y_true.append(y_true_matrix)
                    all_train_y_pred.append(y_pred_matrix)
                    
                    print(f"  文件: {file_name} -> Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
                except Exception as e:
                    print(f"  ✗ 计算指标失败: {e}")

    # 5. 汇总训练集指标
    if all_train_y_true:
        final_y_true = np.vstack(all_train_y_true)
        final_y_pred = np.vstack(all_train_y_pred)
        total_metrics = calculate_metrics(final_y_true, final_y_pred)
        
        print("\n" + "="*40)
        print("训练集汇总评估报告 (只使用通道8)")
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
        report_path = RESULTS_DIR / f"hdc_train_report_channel8_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HDC 训练集评估报告 (只使用通道8)\n")
            f.write(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    try:
        test_channel8_only()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
