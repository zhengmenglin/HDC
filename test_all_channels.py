#!/usr/bin/env python3
"""使用9个通道进行训练和识别的实验"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# 添加torchhd库的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

# 导入超声波数据转换器
from ultrasonic_data_converter import UltrasonicDataConverter
# 导入坐标转换工具
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.coordinate_converter import converter
import json

# ======================================
# 设备设置
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================================
# 配置参数
# ======================================
# 超向量维度
DIMENSIONS = 10000
# 水平嵌入的级别数
NUM_LEVELS = 255
# 批处理大小
BATCH_SIZE = 1
# 通道数，超声波传感器的通道数量
CHANNELS = 9

# 默认参数（可以从命令行覆盖）
DEFAULT_WINDOW = 32
DEFAULT_STEP = 1
DEFAULT_N_GRAM_SIZE = 15

# 窗口大小
WINDOW = DEFAULT_WINDOW
# 滑动窗口步长
STEP = DEFAULT_STEP
# N-gram大小
N_GRAM_SIZE = DEFAULT_N_GRAM_SIZE

# 数据目录
TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
TEST_DATA_DIR = Path(r"d:\HDC\data\test")
# 模型保存路径
MODEL_PATH = Path(r"d:\HDC\models\hdc_model_all_channels.pth")
MODEL_PATH.parent.mkdir(exist_ok=True)

# ======================================
# 9通道数据集类
# ======================================
class AllChannelsDataset(data.Dataset):
    """使用9个通道的数据集类
    
    Args:
        data_dir: 数据目录路径
        is_train: 是否为训练模式
    """
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []
        self.labels = []
        self.file_info = []
        
        self._load_all_data()
    
    def _load_all_data(self):
        """加载所有数据并生成样本"""
        print(f"正在加载 {self.data_dir} 目录下的所有CSV文件")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            print(f"\n正在处理文件: {csv_file.name}")
            
            df = pd.read_csv(csv_file)
            df = df.sort_values(by='encoding')
            
            # 提取9个通道的数据
            channel_data = df[[f'channel_{i}' for i in range(CHANNELS)]].values
            
            # 检查是否有多通道标签
            if all([f'label_{i}' in df.columns for i in range(CHANNELS)]):
                # 使用多通道标签
                labels = df[[f'label_{i}' for i in range(CHANNELS)]].values
                print(f"使用多通道标签，形状: {labels.shape}")
            else:
                # 使用单通道标签（向后兼容）
                labels = df['label'].values if 'label' in df.columns else np.zeros(len(df))
                # 扩展为多通道标签格式
                labels = np.tile(labels[:, np.newaxis], (1, CHANNELS))
                print(f"使用单通道标签并扩展为多通道格式，形状: {labels.shape}")
            
            encodings = df['encoding'].values
            print(f"从 {csv_file.name} 加载了 {len(encodings)} 个编码值")
            
            if len(encodings) < WINDOW:
                print(f"警告: 数据不足，无法生成窗口。需要 {WINDOW} 个编码值，仅获得 {len(encodings)} 个")
                continue
            
            # 生成滑动窗口样本
            file_samples = []
            file_labels = []
            file_indices = []
            
            for start in range(0, len(encodings) - WINDOW + 1, STEP):
                end = start + WINDOW
                
                # 提取窗口内的通道数据，形状为 [window_size, channels]
                window_data = channel_data[start:end]
                sample = window_data.astype(np.float32)
                
                # 确定窗口标签，使用窗口内出现最多的标签
                window_labels = labels[start:end]
                if len(window_labels) > 0:
                    if window_labels.ndim == 2:  # 多通道标签
                        # 对每个通道分别计算出现最多的标签
                        label = []
                        for channel in range(window_labels.shape[1]):
                            channel_labels = window_labels[:, channel]
                            channel_label = max(set(channel_labels), key=list(channel_labels).count)
                            label.append(channel_label)
                        label = np.array(label)
                    else:  # 单通道标签
                        label = max(set(window_labels), key=list(window_labels).count)
                else:
                    label = np.zeros(CHANNELS) if labels.ndim == 2 else 0
                
                file_samples.append(sample)
                file_labels.append(label)
                file_indices.append((str(csv_file), start, end))
            
            self.samples.extend(file_samples)
            self.labels.extend(file_labels)
            self.file_info.extend(file_indices)
            
            print(f"从 {csv_file.name} 生成了 {len(file_samples)} 个窗口样本")
        
        print(f"\n所有文件处理完成，共生成 {len(self.samples)} 个窗口样本")
        if self.labels:
            if isinstance(self.labels[0], np.ndarray):
                # 多通道标签，分别打印每个通道的分布
                for channel in range(self.labels[0].shape[0]):
                    channel_labels = [label[channel] for label in self.labels]
                    print(f"通道 {channel} 标签分布: {np.bincount(channel_labels)}")
            else:
                # 单通道标签
                print(f"标签分布: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        sample = torch.from_numpy(sample)
        return sample, label

# ======================================
# 超维编码器类（多通道版本）
# ======================================
class AllChannelsEncoder(nn.Module):
    """多通道超维编码器
    
    Args:
        out_features: 输出超维向量的维度
        timestamps: 输入时序数据的时间戳数量
        channels: 输入数据的通道数量
        n_gram_size: n-gram大小
    """
    def __init__(self, out_features, timestamps, channels, n_gram_size=N_GRAM_SIZE):
        super(AllChannelsEncoder, self).__init__()
        
        # 通道嵌入层，将每个通道映射到超维空间
        self.channels = embeddings.Random(channels, out_features)
        # 信号嵌入层，将连续信号值映射到超维空间
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=200)
        # n-gram大小
        self.n_gram_size = n_gram_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            input: 输入数据，形状为 [batch, timestamps, channels]
            
        Returns:
            超维向量，形状为 [batch, out_features]
        """
        # 将连续信号嵌入到超维空间
        signal = self.signals(input)
        # 将通道信息与信号嵌入结合
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        
        # 使用multiset聚合样本
        samples = torchhd.multiset(samples)
        # 使用n-gram捕获时序关系
        sample_hv = torchhd.ngrams(samples, n=self.n_gram_size)
        # 硬量化
        return torchhd.hard_quantize(sample_hv)

# ======================================
# 训练函数
# ======================================
def train_all_channels_model(n_gram_size=N_GRAM_SIZE):
    """使用9个通道的数据训练HDC模型
    
    Args:
        n_gram_size: N-gram大小
    """
    print(f"开始训练9通道模型... (N_GRAM_SIZE: {n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP})")
    
    train_ds = AllChannelsDataset(TRAIN_DATA_DIR, is_train=True)
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    if len(train_ds) == 0:
        print("错误: 训练数据集为空！")
        return False
    
    # 检查标签是否为多通道
    is_multichannel = isinstance(train_ds.labels[0], np.ndarray)
    print(f"检测到 {'多通道' if is_multichannel else '单通道'} 标签")
    
    # 获取所有唯一标签，确保模型包含所有可能的类别
    all_labels = []
    for label in train_ds.labels:
        if is_multichannel:
            all_labels.extend(label)
        else:
            all_labels.append(label)
    
    max_label = max(all_labels)
    num_classes = int(max_label + 1)
    print(f"检测到 {num_classes} 个唯一类别（标签 0 到 {max_label}）")
    
    # 初始化编码器
    sample_shape = train_ds[0][0].shape
    encode = AllChannelsEncoder(DIMENSIONS, sample_shape[0], sample_shape[1], n_gram_size=n_gram_size)
    encode = encode.to(device)
    
    # 初始化模型，使用多个Centroid（质心）模型，每个通道一个
    models = []
    if is_multichannel:
        for channel in range(CHANNELS):
            model = Centroid(DIMENSIONS, num_classes)
            model = model.to(device)
            models.append(model)
    else:
        # 单通道标签，只需要一个模型
        model = Centroid(DIMENSIONS, num_classes)
        model = model.to(device)
        models.append(model)
    
    # 训练模型
    print("正在训练模型...")
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="训练中"):
            samples = samples.to(device)
            targets = targets.to(device)
            
            # 生成超维向量
            sample_hv = encode(samples)
            
            if is_multichannel:
                # 多通道标签，为每个通道更新对应的模型
                for channel in range(CHANNELS):
                    channel_targets = targets[:, channel].long()
                    models[channel].add(sample_hv, channel_targets)
            else:
                # 单通道标签，只需要更新一个模型
                models[0].add(sample_hv, targets.long())
    
    # 归一化模型
    for model in models:
        model.normalize()
    
    # 保存模型和编码器
    state_dict = {
        'encoder_state_dict': encode.state_dict(),
        'model_state_dicts': [model.state_dict() for model in models],
        'num_classes': num_classes,
        'sample_shape': sample_shape,
        'n_gram_size': n_gram_size,
        'is_multichannel': is_multichannel
    }
    torch.save(state_dict, MODEL_PATH)
    
    print(f"模型训练完成，已保存到: {MODEL_PATH}")
    return True

# ======================================
# 预测函数
# ======================================
def predict_all_channels_model(data_dir=TEST_DATA_DIR):
    """使用训练好的模型对数据进行预测
    
    Args:
        data_dir: 预测数据目录
    """
    print("开始模型预测...")
    
    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return False
    
    # 加载模型和编码器
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    
    # 获取模型训练时的N_GRAM_SIZE
    n_gram_size = checkpoint.get('n_gram_size', N_GRAM_SIZE)
    is_multichannel = checkpoint.get('is_multichannel', False)
    print(f"预测时使用模型训练参数: N_GRAM_SIZE={n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP}")
    print(f"模型支持 {'多通道' if is_multichannel else '单通道'} 标签")
    
    # 初始化编码器
    sample_shape = checkpoint['sample_shape']
    encode = AllChannelsEncoder(DIMENSIONS, sample_shape[0], sample_shape[1], n_gram_size=n_gram_size)
    encode.load_state_dict(checkpoint['encoder_state_dict'])
    encode = encode.to(device)
    encode.eval()
    
    num_classes = checkpoint['num_classes']
    
    # 初始化多个模型，每个通道一个
    models = []
    model_state_dicts = checkpoint['model_state_dicts']
    for state_dict in model_state_dicts:
        model = Centroid(DIMENSIONS, num_classes)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    
    # 加载数据集
    test_ds = AllChannelsDataset(data_dir, is_train=False)
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(test_ds) == 0:
        print(f"错误: {data_dir} 目录下的数据集为空！")
        return False
    
    # 执行预测
    print(f"正在对 {data_dir} 目录下的数据进行预测...")
    predictions = []
    with torch.no_grad():
        for samples, targets in tqdm(test_ld, desc="预测中"):
            samples = samples.to(device)
            
            # 生成超维向量
            sample_hv = encode(samples)
            
            if is_multichannel:
                # 多通道预测，为每个通道生成预测结果
                pred_labels = []
                for channel in range(CHANNELS):
                    # 使用对应通道的模型进行预测
                    output = models[channel](sample_hv, dot=True)
                    # 获取预测标签
                    pred_label = torch.argmax(output, dim=1).cpu().item()
                    pred_labels.append(pred_label)
                predictions.append(pred_labels)
            else:
                # 单通道预测
                output = models[0](sample_hv, dot=True)
                pred_label = torch.argmax(output, dim=1).cpu().item()
                predictions.append(pred_label)
    
    # 保存预测结果
    print("正在保存预测结果...")
    
    # 按文件分组预测结果
    file_predictions = {}
    for idx, pred_label in enumerate(predictions):
        file_path, start, end = test_ds.file_info[idx]
        file_key = Path(file_path).name
        if file_key not in file_predictions:
            file_predictions[file_key] = []
        file_predictions[file_key].append((start, end, pred_label))
    
    # 为每个文件生成预测结果CSV
    for file_name, pred_info in file_predictions.items():
        try:
            # 读取原始文件
            original_file = data_dir / file_name
            df = pd.read_csv(original_file)
            
            # 添加预测结果列
            if is_multichannel:
                # 多通道预测结果，添加predicted_label_0到predicted_label_8列
                for channel in range(CHANNELS):
                    df[f'predicted_label_{channel}'] = df[f'label_{channel}'] if f'label_{channel}' in df.columns else 0
            else:
                # 单通道预测结果，添加predicted_label列
                df['predicted_label'] = df['label'] if 'label' in df.columns else 0
            
            # 根据预测结果更新标签
            for start, end, pred_label in pred_info:
                if is_multichannel:
                    # 多通道预测结果
                    for channel in range(CHANNELS):
                        df.loc[start:end-1, f'predicted_label_{channel}'] = pred_label[channel]
                else:
                    # 单通道预测结果
                    df.loc[start:end-1, 'predicted_label'] = pred_label
            
            # 保存带预测结果的CSV文件
            output_file = data_dir / f"{original_file.stem.replace('_hdc', '')}_hdc.csv"
            df.to_csv(output_file, index=False)
            print(f"已保存预测结果到: {output_file}")
        except Exception as e:
            print(f"警告: 处理文件 {file_name} 时出错: {e}")
    
    print("所有预测结果已保存")
    return True

# ======================================
# 准确率计算函数
# ======================================
def calculate_all_channels_accuracy(predicted_file, original_file):
    """计算9个通道的识别正确率
    
    计算精确率、召回率和F1分数：
    - 精确率 = 预测正确的点数 / 所有预测出的点数
    - 召回率 = 预测正确的点数 / 所有真实伤损点数
    - F1分数 = 2 * 精确率 * 召回率 / (精确率 + 召回率)
    
    Args:
        predicted_file: 包含预测结果的CSV文件路径
        original_file: 原始CSV文件路径（包含真实标签）
    
    Returns:
        dict: 包含每个通道的精确率、召回率和F1分数的字典
    """
    try:
        pred_df = pd.read_csv(predicted_file)
        orig_df = pd.read_csv(original_file)
        
        metrics = {}
        
        # 检查是否为多通道标签
        is_multichannel = all([f'label_{i}' in orig_df.columns for i in range(CHANNELS)])
        
        if is_multichannel:
            # 多通道标签，分别计算每个通道的准确率
            for channel in range(CHANNELS):
                label_col = f'label_{channel}'
                pred_label_col = f'predicted_label_{channel}'
                
                if label_col not in orig_df.columns or pred_label_col not in pred_df.columns:
                    print(f"警告: 缺少{label_col}或{pred_label_col}列，无法计算通道{channel}的准确率")
                    metrics[f'channel_{channel}'] = {
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0,
                        'total_identified': 0, 'total_true_defects': 0, 'total_correct': 0
                    }
                    continue
                
                if len(pred_df) != len(orig_df):
                    print(f"警告: 预测结果和原始文件行数不一致，无法计算通道{channel}的准确率")
                    metrics[f'channel_{channel}'] = {
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0,
                        'total_identified': 0, 'total_true_defects': 0, 'total_correct': 0
                    }
                    continue
                
                merged_df = pd.merge(orig_df, pred_df, on='encoding', suffixes=('_orig', '_pred'))
                
                if merged_df.empty:
                    print(f"警告: 合并后的数据为空，无法计算通道{channel}的准确率")
                    metrics[f'channel_{channel}'] = {
                        'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0,
                        'total_identified': 0, 'total_true_defects': 0, 'total_correct': 0
                    }
                    continue
                
                true_labels = merged_df[f'{label_col}_orig'].values
                pred_labels = merged_df[f'{pred_label_col}_pred'].values
                
                # 计算各种统计量
                total_identified = np.sum(pred_labels != 0)
                total_true_defects = np.sum(true_labels != 0)
                total_correct = np.sum((pred_labels != 0) & (true_labels != 0))
                
                # 计算精确率
                if total_identified == 0:
                    precision = 0.0
                else:
                    precision = total_correct / total_identified
                
                # 计算召回率
                if total_true_defects == 0:
                    recall = 0.0
                else:
                    recall = total_correct / total_true_defects
                
                # 计算F1分数
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                
                # 计算F2分数
                if (4 * precision + recall) == 0:
                    f2 = 0.0
                else:
                    f2 = 5 * precision * recall / (4 * precision + recall)
                
                metrics[f'channel_{channel}'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'f2': f2,
                    'total_identified': total_identified,
                    'total_true_defects': total_true_defects,
                    'total_correct': total_correct
                }
        else:
            # 单通道标签
            if 'label' not in orig_df.columns or 'predicted_label' not in pred_df.columns:
                print(f"警告: 缺少label或predicted_label列，无法计算准确率")
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0}
            
            if len(pred_df) != len(orig_df):
                print(f"警告: 预测结果和原始文件行数不一致，无法计算准确率")
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0}
            
            merged_df = pd.merge(orig_df, pred_df, on='encoding', suffixes=('_orig', '_pred'))
            
            if merged_df.empty:
                print(f"警告: 合并后的数据为空，无法计算准确率")
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'f2': 0.0}
            
            true_labels = merged_df['label_orig'].values
            pred_labels = merged_df['predicted_label_pred'].values
            
            # 计算各种统计量
            total_identified = np.sum(pred_labels != 0)
            total_true_defects = np.sum(true_labels != 0)
            total_correct = np.sum((pred_labels != 0) & (true_labels != 0))
            
            # 计算精确率
            if total_identified == 0:
                precision = 0.0
            else:
                precision = total_correct / total_identified
            
            # 计算召回率
            if total_true_defects == 0:
                recall = 0.0
            else:
                recall = total_correct / total_true_defects
            
            # 计算F1分数
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            # 计算F2分数
            if (4 * precision + recall) == 0:
                f2 = 0.0
            else:
                f2 = 5 * precision * recall / (4 * precision + recall)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f2': f2,
                'total_identified': total_identified,
                'total_true_defects': total_true_defects,
                'total_correct': total_correct
            }
        
        return metrics
    except Exception as e:
        print(f"计算准确率时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'precision': 0.0, 
            'recall': 0.0, 
            'f1': 0.0,
            'f2': 0.0,
            'total_identified': 0,
            'total_true_defects': 0,
            'total_correct': 0
        }

# ======================================
# 拼孔值提取函数
# ======================================
def extract_beam_parameters(data_dir: Path) -> tuple:
    """从原始JSON文件中提取超声波参数
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        beam_offset_map: 通道到拼孔值的映射
        beam_way_map: 通道到射向的映射（-1后、0直、1前）
        beam_angle_map: 通道到声束角度的映射
    """
    beam_offset_map = {}
    beam_way_map = {}
    beam_angle_map = {}
    
    # 查找所有JSON文件
    json_files = list(data_dir.glob("*.json"))
    json_files = [f for f in json_files if '标注' not in f.name]
    
    if not json_files:
        print(f"警告: {data_dir} 目录下没有找到JSON文件")
        return beam_offset_map, beam_way_map, beam_angle_map
    
    # 使用第一个JSON文件提取参数
    json_file = json_files[0]
    print(f"从 {json_file.name} 提取拼孔参数")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取配置信息
        if data and isinstance(data[0], dict):
            config_item = data[0]
            
            # 提取BeamOffset
            if 'E' in config_item and isinstance(config_item['E'], int) and config_item['E'] == 1:
                if 'BeamOffset' in config_item and isinstance(config_item['BeamOffset'], list):
                    beam_offset = config_item['BeamOffset']
                    for i in range(0, len(beam_offset), 2):
                        if i + 1 < len(beam_offset):
                            channel = int(beam_offset[i])
                            offset = beam_offset[i + 1]
                            beam_offset_map[channel] = offset
            
            # 提取BeamWay
            if 'BeamWay' in config_item and isinstance(config_item['BeamWay'], list):
                beam_way = config_item['BeamWay']
                for i in range(0, len(beam_way), 3):
                    if i + 2 < len(beam_way):
                        channel = int(beam_way[i])
                        way = int(beam_way[i + 1])
                        beam_way_map[channel] = way
            
            # 提取BeamAngle
            if 'BeamAngle' in config_item and isinstance(config_item['BeamAngle'], list):
                beam_angle = config_item['BeamAngle']
                for i in range(0, len(beam_angle), 2):
                    if i + 1 < len(beam_angle):
                        channel = int(beam_angle[i])
                        angle = float(beam_angle[i + 1])
                        beam_angle_map[channel] = angle
        
        print(f"拼孔值提取完成:")
        print(f"  BeamOffset: {beam_offset_map}")
        print(f"  BeamWay: {beam_way_map}")
        print(f"  BeamAngle: {beam_angle_map}")
        
    except Exception as e:
        print(f"提取拼孔参数时出错: {e}")
    
    return beam_offset_map, beam_way_map, beam_angle_map

# ======================================
# 清理旧文件
# ======================================
def clean_old_files(data_dirs):
    """清理旧的hdc.json和hdc.csv文件"""
    for data_dir in data_dirs:
        print(f"清理目录: {data_dir}")
        for file_path in data_dir.glob("*_hdc.csv"):
            try:
                file_path.unlink()
                print(f"✓ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path.name}: {e}")
        for file_path in data_dir.glob("*_hdc.json"):
            try:
                file_path.unlink()
                print(f"✓ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path.name}: {e}")

# ======================================
# 数据转换
# ======================================
def get_data_files(data_dir):
    """获取数据目录下的所有数据文件"""
    data_files = []
    for ext in ['*.json', '*.xxc.json']:
        data_files.extend(data_dir.glob(ext))
    return [f for f in data_files if '标注' not in f.name]

def convert_data_files(data_files, converter):
    """转换数据文件为CSV格式"""
    csv_files = []
    for data_file in data_files:
        annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
        annotation_path = str(annotation_file) if annotation_file.exists() else None
        output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
        
        try:
            result_file = converter.convert_single_file(str(data_file), annotation_path, str(output_csv))
            csv_files.append(str(output_csv))
            print(f"✓ 转换完成: {data_file.name} {'(使用标注文件)' if annotation_path else '(无标注文件)'}")
        except Exception as e:
            print(f"✗ 转换失败 {data_file.name}: {e}")
            continue
    return csv_files

# ======================================
# 主测试函数
# ======================================
def test_all_channels():
    """测试使用9个通道的完整流程"""
    print("=== 开始9通道实验 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    OUTPUT_DIR = Path("all_channels_test_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 清理旧文件
    print("\n=== 清理旧文件 ===")
    clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR])
    print("=== 清理完成 ===")
    
    try:
        # 1. 数据转换
        print("\n=== Step 1: 数据转换 ===")
        converter = UltrasonicDataConverter()
        
        train_files = get_data_files(TRAIN_DATA_DIR)
        train_csv_files = convert_data_files(train_files, converter)
        
        if not train_csv_files:
            print("[FAIL] 没有成功转换的训练数据文件")
            return False
        
        # 提取拼孔值
        print("\n=== 提取拼孔值 ===")
        beam_offset_map, beam_way_map, beam_angle_map = extract_beam_parameters(TRAIN_DATA_DIR)
        
        # 2. 模型训练
        print("\n=== Step 2: 模型训练 ===")
        try:
            if not train_all_channels_model():
                print("[FAIL] 模型训练失败")
                return False
            print("✓ 模型训练成功")
        except Exception as e:
            print(f"[FAIL] 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. 模型预测
        print("\n=== Step 3: 模型预测 ===")
        print("预测训练数据...")
        try:
            if not predict_all_channels_model(TRAIN_DATA_DIR):
                print("[FAIL] 模型预测失败")
                return False
            print("✓ 训练数据预测完成")
        except Exception as e:
            print(f"[FAIL] 模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3.5 CSV到HDC JSON转换
        print("\n=== Step 3.5: CSV到HDC JSON转换 ===")
        try:
            from csv_to_hdc_json import CsvToHdcJsonConverter
            converter = CsvToHdcJsonConverter()
            # 传递拼孔值映射参数
            converter.batch_convert(
                str(TRAIN_DATA_DIR), 
                target_channel=None,  # 所有通道
                beam_offset_map=beam_offset_map,
                beam_way_map=beam_way_map,
                beam_angle_map=beam_angle_map
            )
            print("✓ CSV到HDC JSON转换完成（预测点已添加source标识）")
        except Exception as e:
            print(f"[FAIL] CSV到HDC JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3.6 转换test数据
        print("\n=== Step 3.6: 转换test数据 ===")
        try:
            ultrasonic_converter = UltrasonicDataConverter()
            test_files = get_data_files(TEST_DATA_DIR)
            test_csv_files = convert_data_files(test_files, ultrasonic_converter)
            
            if not test_csv_files:
                print("[WARN] 没有成功转换的test数据文件")
            else:
                print(f"✓ 成功转换 {len(test_csv_files)} 个test数据文件")
        except Exception as e:
            print(f"[FAIL] test数据转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 提取test数据的拼孔值
        print("\n=== 提取test数据拼孔值 ===")
        test_beam_offset_map, test_beam_way_map, test_beam_angle_map = extract_beam_parameters(TEST_DATA_DIR)
        
        # 3.7 预测test数据
        print("\n=== Step 3.7: 预测test数据 ===")
        try:
            if not predict_all_channels_model(TEST_DATA_DIR):
                print("[FAIL] test数据预测失败")
                return False
            print("✓ test数据预测完成")
        except Exception as e:
            print(f"[FAIL] test数据预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3.8 test数据CSV到HDC JSON转换
        print("\n=== Step 3.8: test数据CSV到HDC JSON转换 ===")
        try:
            from csv_to_hdc_json import CsvToHdcJsonConverter
            converter = CsvToHdcJsonConverter()
            # 传递test数据的拼孔值映射参数
            converter.batch_convert(
                str(TEST_DATA_DIR), 
                target_channel=None,  # 所有通道
                beam_offset_map=test_beam_offset_map,
                beam_way_map=test_beam_way_map,
                beam_angle_map=test_beam_angle_map
            )
            print("✓ test数据CSV到HDC JSON转换完成（预测点已添加source标识）")
        except Exception as e:
            print(f"[FAIL] test数据CSV到HDC JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 计算准确率
        print("\n=== Step 4: 计算准确率 ===")
        
        # 检查是否为多通道标签
        is_multichannel = False
        if train_csv_files:
            test_df = pd.read_csv(train_csv_files[0])
            is_multichannel = all([f'label_{i}' in test_df.columns for i in range(CHANNELS)])
        
        if is_multichannel:
            # 多通道标签，分别计算每个通道的准确率
            total_precision = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            total_recall = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            total_f1 = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            total_f2 = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            file_count = {f'channel_{i}': 0 for i in range(CHANNELS)}
            
            for data_file in train_files:
                if '标注' in data_file.name:
                    continue
                
                original_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
                predicted_csv = data_file.parent / f"{data_file.stem.replace('_hdc', '')}_hdc.csv"
                
                if not original_csv.exists() or not predicted_csv.exists():
                    print(f"✗ 缺少文件，无法计算 {data_file.name} 的准确率")
                    continue
                
                metrics = calculate_all_channels_accuracy(predicted_csv, original_csv)
                
                for channel in range(CHANNELS):
                    channel_key = f'channel_{channel}'
                    if channel_key in metrics:
                        total_precision[channel_key] += metrics[channel_key]['precision']
                        total_recall[channel_key] += metrics[channel_key]['recall']
                        total_f1[channel_key] += metrics[channel_key]['f1']
                        total_f2[channel_key] += metrics[channel_key]['f2']
                        file_count[channel_key] += 1
            
            # 计算平均值
            avg_precision = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            avg_recall = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            avg_f1 = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            avg_f2 = {f'channel_{i}': 0.0 for i in range(CHANNELS)}
            
            for channel in range(CHANNELS):
                channel_key = f'channel_{channel}'
                if file_count[channel_key] > 0:
                    avg_precision[channel_key] = total_precision[channel_key] / file_count[channel_key]
                    avg_recall[channel_key] = total_recall[channel_key] / file_count[channel_key]
                    avg_f1[channel_key] = total_f1[channel_key] / file_count[channel_key]
                    avg_f2[channel_key] = total_f2[channel_key] / file_count[channel_key]
            
            # 输出每个通道的准确率
            for channel in range(CHANNELS):
                channel_key = f'channel_{channel}'
                print(f"\n通道 {channel}:")
                print(f"  平均精确率: {avg_precision[channel_key]:.4f}")
                print(f"  平均召回率: {avg_recall[channel_key]:.4f}")
                print(f"  平均F1分数: {avg_f1[channel_key]:.4f}")
                print(f"  平均F2分数: {avg_f2[channel_key]:.4f}")
            
            # 计算总体平均值
            overall_precision = np.mean([avg_precision[f'channel_{i}'] for i in range(CHANNELS)])
            overall_recall = np.mean([avg_recall[f'channel_{i}'] for i in range(CHANNELS)])
            overall_f1 = np.mean([avg_f1[f'channel_{i}'] for i in range(CHANNELS)])
            overall_f2 = np.mean([avg_f2[f'channel_{i}'] for i in range(CHANNELS)])
            
            print(f"\n总体平均:")
            print(f"  平均精确率: {overall_precision:.4f}")
            print(f"  平均召回率: {overall_recall:.4f}")
            print(f"  平均F1分数: {overall_f1:.4f}")
            print(f"  平均F2分数: {overall_f2:.4f}")
        else:
            # 单通道标签
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            total_f2 = 0.0
            file_count = 0
            
            for data_file in train_files:
                if '标注' in data_file.name:
                    continue
                
                original_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
                predicted_csv = data_file.parent / f"{data_file.stem.replace('_hdc', '')}_hdc.csv"
                
                if not original_csv.exists() or not predicted_csv.exists():
                    print(f"✗ 缺少文件，无法计算 {data_file.name} 的准确率")
                    continue
                
                metrics = calculate_all_channels_accuracy(predicted_csv, original_csv)
                total_precision += metrics['precision']
                total_recall += metrics['recall']
                total_f1 += metrics['f1']
                total_f2 += metrics['f2']
                file_count += 1
            
            avg_precision = total_precision / file_count if file_count > 0 else 0.0
            avg_recall = total_recall / file_count if file_count > 0 else 0.0
            avg_f1 = total_f1 / file_count if file_count > 0 else 0.0
            avg_f2 = total_f2 / file_count if file_count > 0 else 0.0
            
            print(f"\n平均精确率: {avg_precision:.4f}")
            print(f"平均召回率: {avg_recall:.4f}")
            print(f"平均F1分数: {avg_f1:.4f}")
            print(f"平均F2分数: {avg_f2:.4f}")
        
        # 5. 输出结果
        print("\n" + "="*50)
        print("=== 9通道实验结果 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"使用的窗口大小: {WINDOW}")
        print(f"使用的滑动步长: {STEP}")
        print(f"使用的N_GRAM_SIZE: {N_GRAM_SIZE}")
        print(f"使用的通道数: {CHANNELS}")
        print("="*50)
        
        # 保存结果
        results_df = pd.DataFrame([{
            'window': WINDOW,
            'step': STEP,
            'n_gram': N_GRAM_SIZE,
            'channels': CHANNELS,
            'precision': overall_precision if is_multichannel else avg_precision,
            'recall': overall_recall if is_multichannel else avg_recall,
            'f1': overall_f1 if is_multichannel else avg_f1,
            'f2': overall_f2 if is_multichannel else avg_f2
        }])
        results_file = OUTPUT_DIR / "all_channels_accuracy_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n测试结果已保存到: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='9通道HDC实验')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW, 
                       help=f'窗口大小（默认: {DEFAULT_WINDOW}）')
    parser.add_argument('--step', type=int, default=DEFAULT_STEP, 
                       help=f'滑动步长（默认: {DEFAULT_STEP}）')
    parser.add_argument('--n-gram', type=int, default=DEFAULT_N_GRAM_SIZE, 
                       help=f'N-gram大小（默认: {DEFAULT_N_GRAM_SIZE}）')
    parser.add_argument('--no-convert', action='store_true', 
                       help='跳过数据转换，直接使用已有的CSV文件')
    
    args = parser.parse_args()
    
    # 更新全局参数
    global WINDOW, STEP, N_GRAM_SIZE
    WINDOW = args.window
    STEP = args.step
    N_GRAM_SIZE = args.n_gram
    
    print("=== 9通道HDC实验 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用的窗口大小: {WINDOW}")
    print(f"使用的滑动步长: {STEP}")
    print(f"使用的N_GRAM_SIZE: {N_GRAM_SIZE}")
    print(f"使用的通道数: {CHANNELS}")
    
    success = test_all_channels()
    
    if success:
        print("\n✓ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
