"""
超声波探伤HDC分类器
基于torchhd库实现，参考EMG手势识别的数据处理方式
"""

import torch
import torch.nn as nn
import torchhd
from torchhd import embeddings
import numpy as np
import json
import pandas as pd
from typing import List, Tuple, Dict, Optional
import math
from pathlib import Path


class UltrasonicHDCEncoder(nn.Module):
    """
    超声波HDC编码器
    处理9通道超声波数据，将时序信号编码为超向量
    """

    def __init__(self, out_features=10000, channels=9, window_size=256):
        super().__init__()
        self.out_features = out_features
        self.channels = channels
        self.window_size = window_size

        # 为9个超声波通道创建嵌入层
        self.channel_embeds = nn.ModuleList([
            embeddings.Random(window_size, out_features)
            for _ in range(channels)
        ])

        # 深度、幅度、dB值编码器
        self.depth_embed = embeddings.Level(256, out_features, low=0, high=300)    # 深度0-300mm
        self.amplitude_embed = embeddings.Level(101, out_features, low=0, high=100) # 幅度0-100%
        self.db_embed = embeddings.Level(101, out_features, low=-60, high=0)       # dB值-60到0

    def forward(self, channel_data):
        """
        前向传播：将超声波数据编码为超向量

        Args:
            channel_data: [batch, channels, window_size, features]
                         features顺序: [depth, amplitude, db_value]

        Returns:
            sample_hv: 编码后的超向量 [batch, out_features]
        """
        batch_size = channel_data.shape[0]
        channel_hvs = []

        for i in range(self.channels):  # 9个超声波通道
            # 提取三个特征序列
            depth_seq = channel_data[:, i, :, 0]      # 深度序列 [batch, window_size]
            amplitude_seq = channel_data[:, i, :, 1]  # 幅度序列 [batch, window_size]
            db_seq = channel_data[:, i, :, 2]         # dB序列 [batch, window_size]

            # 编码为超向量
            depth_hv = self.depth_embed(depth_seq)           # [batch, out_features]
            amplitude_hv = self.amplitude_embed(amplitude_seq) # [batch, out_features]
            db_hv = self.db_embed(db_seq)                     # [batch, out_features]

            # 绑定三个特征 (深度 × 幅度 × dB)
            signal_hv = torchhd.bind(
                torchhd.bind(depth_hv, amplitude_hv),
                db_hv
            )

            # N-gram处理时间序列 (类似EMG的n-gram)
            signal_hv = torchhd.ngrams(signal_hv, n=4)
            channel_hvs.append(signal_hv)

        # 多重集整合所有通道 (类似EMG的multiset)
        if len(channel_hvs) == 1:
            sample_hv = channel_hvs[0]
        else:
            # 将通道向量堆叠后进行multiset操作
            channel_stack = torch.stack(channel_hvs, dim=0)
            sample_hv = torchhd.multiset(channel_stack)
        return torchhd.hard_quantize(sample_hv)


class UltrasonicHDCClassifier:
    """
    超声波HDC分类器
    支持11类伤损检测：10种伤损类型 + 1种正常结构
    """

    def __init__(self, encoder=None, d=10000):
        """
        初始化分类器

        Args:
            encoder: 预训练的编码器，如果为None则创建新编码器
            d: 超向量维度
        """
        if encoder is None:
            self.encoder = UltrasonicHDCEncoder(out_features=d)
        else:
            self.encoder = encoder

        self.d = d
        self.class_vectors = {}  # 类别超向量
        self.class_labels = {}   # 类别标签映射

        # 伤损类型映射 (根据需求文档)
        self.flaw_types = {
            0: "正常结构",
            1: "轨头外侧核伤",
            2: "轨头中心核伤",
            3: "轨头内侧核伤",
            4: "轨头内、外、中心核伤",
            5: "轨头内、中心核伤",
            6: "轨头外、中心核伤",
            7: "轨头水平分裂",
            8: "轨头、轨腰分裂",
            9: "轨腰水平裂",
            10: "轨底断裂",
            11: "螺孔裂纹"
        }

        # 正常结构类型映射
        self.normal_types = {
            101: "接头",
            102: "螺孔",
            103: "导线孔",
            104: "焊缝",
            105: "鱼鳞伤",
            106: "轨底锈蚀",
            107: "轨底焊筋",
            108: "道岔"
        }

    def train(self, samples, labels):
        """
        训练HDC分类器

        Args:
            samples: 样本列表，每个样本为[channel_data, window_size, features]
            labels: 对应标签列表 (0-11)
        """
        print(f"开始训练，共 {len(samples)} 个样本...")

        # 按类别分组样本
        class_groups = {}
        for sample, label in zip(samples, labels):
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(sample)

        # 为每个类别生成类别超向量
        for label, sample_list in class_groups.items():
            print(f"处理类别 {label} ({self.flaw_types.get(label, '未知')}): {len(sample_list)} 个样本")

            sample_vectors = []
            for sample in sample_list:
                # 转换为tensor并编码
                sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # [1, channels, window, features]
                sample_vector = self.encoder(sample_tensor).squeeze(0)  # [d]
                sample_vectors.append(sample_vector)

            # 捆绑同类所有样本向量
                if sample_vectors:
                    if len(sample_vectors) == 1:
                        # 只有一个样本时，直接使用该向量
                        class_vector = sample_vectors[0]
                    else:
                        # 多个样本时进行捆绑 - 依次捆绑所有向量
                        class_vector = sample_vectors[0]
                        for i in range(1, len(sample_vectors)):
                            class_vector = torchhd.bundle(class_vector, sample_vectors[i])
                self.class_vectors[label] = class_vector
                self.class_labels[label] = label

        print(f"训练完成，共生成 {len(self.class_vectors)} 个类别向量")

    def predict(self, sample):
        """
        预测单个样本的类别

        Args:
            sample: 样本数据 [channels, window_size, features]

        Returns:
            predicted_label: 预测标签
            similarities: 与各类别相似度字典
        """
        # 如果模型未训练，返回默认值
        if not self.class_vectors:
            return 0, {0: 1.0}  # 默认返回正常结构

        # 编码查询样本
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # [1, channels, window, features]
        query_vector = self.encoder(sample_tensor).squeeze(0)  # [d]

        # 计算与每个类别向量的相似度
        similarities = {}
        for label, class_vector in self.class_vectors.items():
            similarity = torchhd.cosine_similarity(query_vector, class_vector)
            # 取平均值作为相似度，处理多个通道的情况
            similarities[label] = similarity.mean().item()

        # 选择相似度最高的类别
        if similarities:
            predicted_label = max(similarities.items(), key=lambda x: x[1])[0]
        else:
            predicted_label = 0  # 默认值

        return predicted_label, similarities

    def predict_batch(self, samples):
        """
        批量预测样本类别

        Args:
            samples: 样本列表

        Returns:
            predictions: 预测结果列表
        """
        predictions = []
        for sample in samples:
            pred_label, similarities = self.predict(sample)
            predictions.append((pred_label, similarities))

        return predictions

    def save_model(self, filepath):
        """
        保存模型到文件

        Args:
            filepath: 保存路径
        """
        model_data = {
            'encoder_state': self.encoder.state_dict(),
            'd': self.d,
            'class_vectors': {k: v.tolist() for k, v in self.class_vectors.items()},
            'class_labels': self.class_labels,
            'flaw_types': self.flaw_types,
            'normal_types': self.normal_types
        }

        torch.save(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath):
        """
        从文件加载模型

        Args:
            filepath: 模型文件路径
        """
        # 在PyTorch 2.6+中，torch.load默认weights_only=True，需要设置为False来加载包含自定义类型的模型
        model_data = torch.load(filepath, weights_only=False)

        # 恢复编码器状态
        self.encoder = UltrasonicHDCEncoder(out_features=model_data['d'])
        self.encoder.load_state_dict(model_data['encoder_state'])

        # 恢复类别向量
        self.class_vectors = {k: torch.tensor(v) for k, v in model_data['class_vectors'].items()}
        self.class_labels = model_data['class_labels']
        self.flaw_types = model_data['flaw_types']
        self.normal_types = model_data['normal_types']

        print(f"模型已加载: {len(self.class_vectors)} 个类别向量")

    def get_class_name(self, label):
        """
        获取类别标签对应的名称

        Args:
            label: 类别标签

        Returns:
            class_name: 类别名称
        """
        if label in self.flaw_types:
            return self.flaw_types[label]
        elif label in self.normal_types:
            return self.normal_types[label]
        else:
            return f"未知类别_{label}"