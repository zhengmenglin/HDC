"""
超维计算编码器
实现Level编码、循环移位编码等
"""

import numpy as np
import torch
import torchhd
from typing import List, Optional
from config import config

class HDCEncoder:
    """超维计算编码器"""
    
    def __init__(self, dimensions: int = None, vsa_type: str = None):
        self.dimensions = dimensions or config.get("model.dimensions", 10000)
        self.vsa_type = vsa_type or config.get("model.vsa_type", "MAP")
        
        # 创建各种超向量
        self._initialize_vectors()
    
    def _initialize_vectors(self):
        """初始化各类超向量"""
        # 随机超向量
        self.random_hv = torchhd.random(1, self.dimensions, self.vsa_type).squeeze(0)
        
        # 空超向量（bundle操作的恒等元）
        self.empty_hv = torchhd.empty(1, self.dimensions, self.vsa_type).squeeze(0)
        
        # 单位超向量（bind操作的恒等元）
        self.identity_hv = torchhd.identity(1, self.dimensions, self.vsa_type).squeeze(0)
        
        # 为每个深度级别创建Level超向量
        self.depth_levels = 1000
        self.depth_hvs = torchhd.level(
            self.depth_levels,
            self.dimensions,
            self.vsa_type,
            randomness=0.0
        )
        
        # 为每个幅度级别创建超向量
        self.amplitude_levels = 100
        self.amplitude_hvs = torchhd.level(
            self.amplitude_levels,
            self.dimensions,
            self.vsa_type,
            randomness=0.0
        )
        
        # 为每个通道创建随机超向量
        self.channel_hvs = torchhd.random(9, self.dimensions, self.vsa_type)
        
        # 为每个编码值创建随机超向量
        self.encoding_levels = 1000
        self.encoding_hvs = torchhd.random(self.encoding_levels, self.dimensions, self.vsa_type)
    
    def encode_depth_continuous(self, depth: float, min_depth: float = 0.0, 
                              max_depth: float = 200.0) -> torch.Tensor:
        """连续深度值编码（使用Level编码）
        
        Args:
            depth: 深度值
            min_depth: 最小深度
            max_depth: 最大深度
        
        Returns:
            超维向量
        """
        # 将深度值映射到Level索引
        if depth <= min_depth:
            level_idx = 0
        elif depth >= max_depth:
            level_idx = self.depth_levels - 1
        else:
            level_idx = int((depth - min_depth) / (max_depth - min_depth) * (self.depth_levels - 1))
            level_idx = min(level_idx, self.depth_levels - 1)
        
        return self.depth_hvs[level_idx]
    
    def encode_depths(self, depths: np.ndarray, min_depth: float = 0.0,
                    max_depth: float = 200.0) -> torch.Tensor:
        """批量编码深度值
        
        Args:
            depths: 深度值数组
            min_depth: 最小深度
            max_depth: 最大深度
        
        Returns:
            超维向量数组
        """
        hvs = []
        for depth in depths:
            hv = self.encode_depth_continuous(depth, min_depth, max_depth)
            hvs.append(hv)
        
        return torch.stack(hvs, dim=0)
    
    def encode_amplitude_continuous(self, amplitude: float, 
                                 min_amp: float = 0.0,
                                 max_amp: float = 100.0) -> torch.Tensor:
        """连续幅度值编码
        
        Args:
            amplitude: 幅度值
            min_amp: 最小幅度
            max_amp: 最大幅度
        
        Returns:
            超维向量
        """
        if amplitude <= min_amp:
            level_idx = 0
        elif amplitude >= max_amp:
            level_idx = self.amplitude_levels - 1
        else:
            level_idx = int((amplitude - min_amp) / (max_amp - min_amp) * (self.amplitude_levels - 1))
            level_idx = min(level_idx, self.amplitude_levels - 1)
        
        return self.amplitude_hvs[level_idx]
    
    def encode_amplitudes(self, amplitudes: np.ndarray) -> torch.Tensor:
        """批量编码幅度值
        
        Args:
            amplitudes: 幅度值数组
        
        Returns:
            超维向量数组
        """
        hvs = []
        for amp in amplitudes:
            hv = self.encode_amplitude_continuous(amp)
            hvs.append(hv)
        
        return torch.stack(hvs, dim=0)
    
    def encode_channel(self, channel_idx: int) -> torch.Tensor:
        """编码通道号
        
        Args:
            channel_idx: 通道索引 (0-8)
        
        Returns:
            超维向量
        """
        return self.channel_hvs[channel_idx]
    
    def encode_encoding_value(self, encoding_value: int) -> torch.Tensor:
        """编码编码值
        
        Args:
            encoding_value: 编码值
        
        Returns:
            超维向量
        """
        if encoding_value < self.encoding_levels:
            return self.encoding_hvs[encoding_value]
        else:
            # 如果编码值超出范围，使用随机编码
            return self.random_hv
    
    def circular_shift_encode(self, hvs: torch.Tensor) -> torch.Tensor:
        """循环移位编码
        
        将多个超向量通过循环移位融合为一个超向量
        
        Args:
            hvs: 超向量数组，形状为 [n, dimensions]
        
        Returns:
            融合后的超向量，形状为 [dimensions]
        """
        n = hvs.size(0)
        
        if n == 0:
            return self.empty_hv.clone()
        
        if n == 1:
            return hvs[0].clone()
        
        # 循环移位编码
        result = hvs[0].clone()
        for i in range(1, n):
            # 对当前向量进行移位
            shifted = torchhd.permute(hvs[i], shifts=i)
            # 使用bundle操作融合
            result = torchhd.bundle(result, shifted)
        
        return result
    
    def bundle_encode(self, hvs: torch.Tensor) -> torch.Tensor:
        """捆绑编码
        
        将多个超向量捆绑为一个超向量
        
        Args:
            hvs: 超向量数组，形状为 [n, dimensions]
        
        Returns:
            融合后的超向量，形状为 [dimensions]
        """
        return torchhd.multiset(hvs)
    
    def bind_encode(self, hvs: torch.Tensor) -> torch.Tensor:
        """绑定编码
        
        将多个超向量绑定为一个超向量
        
        Args:
            hvs: 超向量数组，形状为 [n, dimensions]
        
        Returns:
            融合后的超向量，形状为 [dimensions]
        """
        return torchhd.multibind(hvs)
    
    def encode_sample(self, sample: dict) -> torch.Tensor:
        """编码单个样本
        
        Args:
            sample: 样本字典，包含 'encoding', 'channel', 'depth', 'amplitude'
        
        Returns:
            超维向量
        """
        # 提取各个特征
        encoding_value = sample.get('encoding', 0)
        channel = sample.get('channel', 0)
        depth = sample.get('depth', 0.0)
        amplitude = sample.get('amplitude', 0.0)
        
        # 编码各个特征
        encoding_hv = self.encode_encoding_value(encoding_value)
        channel_hv = self.encode_channel(channel)
        depth_hv = self.encode_depth_continuous(depth)
        amplitude_hv = self.encode_amplitude_continuous(amplitude)
        
        # 绑定编码值和通道
        step1 = torchhd.bind(encoding_hv, channel_hv)
        
        # 绑定深度和幅度
        step2 = torchhd.bind(depth_hv, amplitude_hv)
        
        # 最终融合
        final_hv = torchhd.bind(step1, step2)
        
        return final_hv
    
    def encode_multi_channel_sample(self, data: np.ndarray) -> torch.Tensor:
        """编码多通道样本
        
        Args:
            data: 数据数组，形状为 [n_samples, 10] (编码值 + 9通道深度值)
        
        Returns:
            超维向量
        """
        encoding_value = int(data[0])
        channel_depths = data[1:10]
        
        # 编码编码值
        encoding_hv = self.encode_encoding_value(encoding_value)
        
        # 编码各个通道
        channel_hvs = []
        for i, depth in enumerate(channel_depths):
            if depth > 0:  # 只编码有效深度值
                depth_hv = self.encode_depth_continuous(depth)
                channel_hv = torchhd.bind(depth_hv, self.channel_hvs[i])
                channel_hvs.append(channel_hv)
        
        if channel_hvs:
            # 融合通道信息
            channels_fused = self.bundle_encode(torch.stack(channel_hvs, dim=0))
            # 最终融合
            final_hv = torchhd.bind(encoding_hv, channels_fused)
        else:
            final_hv = encoding_hv
        
        return final_hv
    
    def encode_sequence(self, samples: List[dict]) -> torch.Tensor:
        """编码序列样本
        
        Args:
            samples: 样本列表
        
        Returns:
            序列超维向量
        """
        if not samples:
            return self.empty_hv.clone()
        
        # 编码每个样本
        sample_hvs = []
        for i, sample in enumerate(samples):
            sample_hv = self.encode_sample(sample)
            # 使用循环移位表示序列中的位置
            if i > 0:
                sample_hv = torchhd.permute(sample_hv, shifts=i)
            sample_hvs.append(sample_hv)
        
        # 融合序列
        sequence_hv = self.bundle_encode(torch.stack(sample_hvs, dim=0))
        
        return sequence_hv
    
    def compute_similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> float:
        """计算两个超向量的相似度
        
        Args:
            hv1: 超向量1
            hv2: 超向量2
        
        Returns:
            相似度值
        """
        return torchhd.cosine_similarity(hv1.unsqueeze(0), hv2.unsqueeze(0)).item()
    
    def normalize_hv(self, hv: torch.Tensor) -> torch.Tensor:
        """归一化超向量
        
        Args:
            hv: 超向量
        
        Returns:
            归一化后的超向量
        """
        return torchhd.normalize(hv.unsqueeze(0)).squeeze(0)
    
    def quantize_hv(self, hv: torch.Tensor) -> torch.Tensor:
        """量化超向量为二进制
        
        Args:
            hv: 超向量
        
        Returns:
            二值化后的超向量
        """
        return hv.sign()
    
    def get_info(self) -> dict:
        """获取编码器信息"""
        return {
            "dimensions": self.dimensions,
            "vsa_type": self.vsa_type,
            "depth_levels": self.depth_levels,
            "amplitude_levels": self.amplitude_levels,
            "encoding_levels": self.encoding_levels,
            "num_channels": 9
        }