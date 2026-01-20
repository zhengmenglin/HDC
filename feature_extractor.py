"""
超声波数据特征提取器
实现通道分离、超维编码、特征融合
"""

import numpy as np
import torch
import torchhd
from typing import List, Tuple, Optional
from config import config

class FeatureExtractor:
    """超声波数据特征提取器"""
    
    def __init__(self, dimensions: int = None, vsa_type: str = "MAP"):
        self.dimensions = dimensions or config.get("model.dimensions", 10000)
        self.vsa_type = vsa_type or config.get("model.vsa_type", "MAP")
        
        # 为每个通道创建Level编码器
        self.level_encoders = [
            torchhd.embeddings.Level(
                num_embeddings=1000,  # 深度值的离散级别数
                embedding_dim=self.dimensions,
                vsa=self.vsa_type,
                low=0.0,
                high=200.0  # 深度值范围
            ) for _ in range(9)
        ]
        
        # 为编码值创建Level编码器
        self.encoding_encoder = torchhd.embeddings.Level(
            num_embeddings=1000,
            embedding_dim=self.dimensions,
            vsa=self.vsa_type,
            low=0.0,
            high=1000.0  # 编码值范围
        )
        
        # 为幅度值创建Level编码器
        self.amplitude_encoder = torchhd.embeddings.Level(
            num_embeddings=100,
            embedding_dim=self.dimensions,
            vsa=self.vsa_type,
            low=0.0,
            high=100.0  # 幅度值范围
        )
    
    def separate_channels(self, data: np.ndarray) -> List[np.ndarray]:
        """分离9个通道的数据
        
        Args:
            data: 输入数据，形状为 [n_samples, 10] (编码值 + 9通道深度值)
        
        Returns:
            9个通道的数据列表，每个形状为 [n_samples]
        """
        channels = []
        for i in range(9):
            channel_data = data[:, i+1]  # 跳过编码值列
            channels.append(channel_data)
        return channels
    
    def encode_channel(self, channel_data: np.ndarray, encoder) -> torch.Tensor:
        """编码单个通道的数据
        
        Args:
            channel_data: 通道数据，形状为 [n_samples]
            encoder: 编码器
        
        Returns:
            超维向量，形状为 [n_samples, dimensions]
        """
        # 将numpy数组转换为torch张量
        data_tensor = torch.from_numpy(channel_data).float()
        
        # 编码为超维向量
        hv = encoder(data_tensor)
        
        return hv
    
    def encode_all_channels(self, data: np.ndarray) -> List[torch.Tensor]:
        """编码所有9个通道的数据
        
        Args:
            data: 输入数据，形状为 [n_samples, 10]
        
        Returns:
            9个通道的超维向量列表
        """
        channels = self.separate_channels(data)
        channel_hvs = []
        
        for i, channel_data in enumerate(channels):
            hv = self.encode_channel(channel_data, self.level_encoders[i])
            channel_hvs.append(hv)
        
        return channel_hvs
    
    def encode_encoding_value(self, encoding_values: np.ndarray) -> torch.Tensor:
        """编码编码值
        
        Args:
            encoding_values: 编码值，形状为 [n_samples]
        
        Returns:
            超维向量，形状为 [n_samples, dimensions]
        """
        data_tensor = torch.from_numpy(encoding_values).float()
        hv = self.encoding_encoder(data_tensor)
        return hv
    
    def encode_amplitude(self, amplitude_values: np.ndarray) -> torch.Tensor:
        """编码幅度值
        
        Args:
            amplitude_values: 幅度值，形状为 [n_samples]
        
        Returns:
            超维向量，形状为 [n_samples, dimensions]
        """
        data_tensor = torch.from_numpy(amplitude_values).float()
        hv = self.amplitude_encoder(data_tensor)
        return hv
    
    def circular_shift_encode(self, hvs: torch.Tensor) -> torch.Tensor:
        """循环移位编码
        
        Args:
            hvs: 超维向量，形状为 [n_samples, dimensions]
        
        Returns:
            融合后的超维向量，形状为 [dimensions]
        """
        # 使用bundle操作融合所有超维向量
        fused_hv = torchhd.multiset(hvs)
        
        return fused_hv
    
    def bundle_channels(self, channel_hvs: List[torch.Tensor]) -> torch.Tensor:
        """捆绑多个通道的超维向量
        
        Args:
            channel_hvs: 9个通道的超维向量列表
        
        Returns:
            捆绑后的超维向量，形状为 [n_samples, dimensions]
        """
        # 将所有通道的超维向量堆叠
        stacked = torch.stack(channel_hvs, dim=1)  # [n_samples, 9, dimensions]
        
        # 使用bundle操作融合
        fused = torchhd.multiset(stacked)
        
        return fused
    
    def encode_sample(self, sample: np.ndarray) -> torch.Tensor:
        """编码单个样本
        
        Args:
            sample: 单个样本，形状为 [10] (编码值 + 9通道深度值)
        
        Returns:
            超维向量，形状为 [dimensions]
        """
        # 提取编码值
        encoding_value = sample[0]
        
        # 提取9个通道的深度值
        channel_depths = sample[1:]
        
        # 编码编码值
        encoding_hv = self.encode_encoding_value(np.array([encoding_value]))
        
        # 编码每个通道
        channel_hvs = []
        for i, depth in enumerate(channel_depths):
            channel_hv = self.encode_channel(np.array([depth]), self.level_encoders[i])
            channel_hvs.append(channel_hv)
        
        # 捆绑所有通道
        channel_fused = torchhd.multiset(torch.stack(channel_hvs, dim=0))
        
        # 绑定编码值和通道融合结果
        final_hv = torchhd.bind(encoding_hv.squeeze(0), channel_fused.squeeze(0))
        
        return final_hv
    
    def encode_batch(self, data: np.ndarray) -> torch.Tensor:
        """批量编码样本
        
        Args:
            data: 输入数据，形状为 [n_samples, 10]
        
        Returns:
            超维向量，形状为 [n_samples, dimensions]
        """
        n_samples = data.shape[0]
        hv_list = []
        
        for i in range(n_samples):
            hv = self.encode_sample(data[i])
            hv_list.append(hv)
        
        # 堆叠所有超维向量
        hv_batch = torch.stack(hv_list, dim=0)
        
        return hv_batch
    
    def extract_features(self, data: np.ndarray, 
                      use_encoding: bool = True,
                      use_channels: bool = True) -> torch.Tensor:
        """提取特征
        
        Args:
            data: 输入数据，形状为 [n_samples, 10]
            use_encoding: 是否使用编码值
            use_channels: 是否使用通道数据
        
        Returns:
            超维向量，形状为 [n_samples, dimensions]
        """
        if use_encoding and use_channels:
            return self.encode_batch(data)
        elif use_channels:
            channel_hvs = self.encode_all_channels(data)
            return self.bundle_channels(channel_hvs)
        else:
            encoding_values = data[:, 0]
            return self.encode_encoding_value(encoding_values)
    
    def create_sequence_encoding(self, data: np.ndarray, 
                               sequence_length: int = 5) -> torch.Tensor:
        """创建序列编码
        
        Args:
            data: 输入数据，形状为 [n_samples, 10]
            sequence_length: 序列长度
        
        Returns:
            序列编码的超维向量
        """
        n_samples = data.shape[0]
        sequence_hvs = []
        
        for i in range(n_samples):
            # 提取序列
            if i + sequence_length <= n_samples:
                sequence_data = data[i:i+sequence_length]
                
                # 编码序列中的每个样本
                sample_hvs = []
                for j in range(sequence_length):
                    sample_hv = self.encode_sample(sequence_data[j])
                    sample_hvs.append(sample_hv)
                
                # 使用循环移位编码序列
                sequence_hv = self.circular_shift_encode(torch.stack(sample_hvs, dim=0))
                sequence_hvs.append(sequence_hv)
        
        if sequence_hvs:
            return torch.stack(sequence_hvs, dim=0)
        else:
            return torch.empty(0, self.dimensions)
    
    def get_feature_info(self) -> dict:
        """获取特征信息"""
        return {
            "dimensions": self.dimensions,
            "vsa_type": self.vsa_type,
            "num_channels": 9,
            "encoding_levels": 1000,
            "depth_range": [0.0, 200.0],
            "amplitude_range": [0.0, 100.0]
        }