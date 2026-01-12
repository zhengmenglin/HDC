import numpy as np
import json
from collections import defaultdict
import random
import math
from typing import Dict, List, Tuple, Optional


def calculate_actual_depth(acoustic_range, horizontal_angle, pitch_angle):
    """计算实际深度值（经角度矫正）"""
    return acoustic_range * math.cos(math.radians(horizontal_angle)) * math.cos(math.radians(pitch_angle))


def calculate_encoding_offset(acoustic_range, horizontal_angle, pitch_angle):
    """计算编码偏差"""
    return acoustic_range * math.cos(math.radians(horizontal_angle)) * math.sin(math.radians(pitch_angle))


class BScanPoint:
    """B显数据点类，存储深度值、原始dB值、幅度值"""
    def __init__(self, depth: float = None, db_value: float = None, amplitude: float = 100.0):
        """
        初始化B显数据点
        
        Args:
            depth: 深度值（mm）
            db_value: 原始超声信号强度（dB）
            amplitude: 信号幅度值（%），默认为100%
        """
        self.depth = depth
        self.db_value = db_value
        self.amplitude = amplitude


class ChannelConfig:
    """通道配置类，存储通道的参数信息"""
    def __init__(self, pitch_angle: float, horizontal_angle: float, offset: float):
        """
        初始化通道配置
        
        Args:
            pitch_angle: 俯仰角度（度）
            horizontal_angle: 水平角度（度）
            offset: 拼孔值（mm）
        """
        self.pitch_angle = pitch_angle
        self.horizontal_angle = horizontal_angle
        self.offset = offset


class ChannelManager:
    """通道管理器，负责解析和管理9个通道的配置"""
    
    def __init__(self, config_data=None):
        self.channels = {}
        # 初始化默认通道配置
        self._init_default_channels()
        # 如果提供了配置数据，则解析并更新
        if config_data:
            self.parse_config(config_data)
    
    def _init_default_channels(self):
        """初始化默认的9个通道配置"""
        # 默认通道配置：(俯仰角度, 水平角度, 拼孔值)
        default_configs = [
            (70, 20, 10),   # 通道1：俯仰70°, 水平角度α(默认20°), 拼孔值10
            (70, 20, 60),   # 通道2：俯仰70°, 水平角度α(默认20°), 拼孔值60
            (70, 20, 20),   # 通道3：俯仰70°, 水平角度α(默认20°), 拼孔值20
            (70, 20, 50),   # 通道4：俯仰70°, 水平角度α(默认20°), 拼孔值50
            (70, 0, 30),    # 通道5：俯仰70°, 水平角度0°, 拼孔值30
            (70, 0, 40),    # 通道6：俯仰70°, 水平角度0°, 拼孔值40
            (37, 0, 0),     # 通道7：俯仰37°, 水平角度0°, 拼孔值0
            (37, 0, 70),    # 通道8：俯仰37°, 水平角度0°, 拼孔值70
            (0, 0, 80)      # 通道9：俯仰0°, 水平角度0°, 拼孔值80
        ]
        
        for i, (pitch, horizontal, offset) in enumerate(default_configs):
            channel_id = i + 1  # 通道号从1开始
            self.channels[channel_id] = ChannelConfig(pitch, horizontal, offset)
    
    def parse_config(self, config_data):
        """从JSON配置数据解析通道参数"""
        if isinstance(config_data, dict):
            # 解析BeamAngle配置（列表格式：[通道号1, 角度1, 通道号2, 角度2, ...]）
            if 'BeamAngle' in config_data and isinstance(config_data['BeamAngle'], list):
                beam_angle_list = config_data['BeamAngle']
                for i in range(0, len(beam_angle_list), 2):
                    if i + 1 < len(beam_angle_list):
                        channel_id = int(beam_angle_list[i])
                        pitch = float(beam_angle_list[i+1])
                        # 更新通道配置
                        if channel_id in self.channels:
                            self.channels[channel_id].pitch_angle = pitch
                            self.channels[channel_id].horizontal_angle = 0.0  # 默认水平角度为0
            
            # 解析BeamOffset配置（列表格式：[通道号1, 偏移1, 通道号2, 偏移2, ...]）
            if 'BeamOffset' in config_data and isinstance(config_data['BeamOffset'], list):
                beam_offset_list = config_data['BeamOffset']
                for i in range(0, len(beam_offset_list), 2):
                    if i + 1 < len(beam_offset_list):
                        channel_id = int(beam_offset_list[i])
                        offset = float(beam_offset_list[i+1])
                        # 更新通道配置
                        if channel_id in self.channels:
                            self.channels[channel_id].offset = offset
            
            # 解析BeamWay配置（列表格式：[通道号1, 射向1, 侧边1, 通道号2, 射向2, 侧边2, ...]）
            if 'BeamWay' in config_data and isinstance(config_data['BeamWay'], list):
                beam_way_list = config_data['BeamWay']
                # 注意：BeamWay格式为每三个数为一组，依次是通道号、射向、侧边
                for i in range(0, len(beam_way_list), 3):
                    if i + 2 < len(beam_way_list):
                        channel_id = int(beam_way_list[i])
                        # 射向和侧边信息暂时不使用，只记录通道号
                        pass
    
    def get_channel_config(self, channel_id):
        """获取指定通道的配置"""
        return self.channels.get(channel_id)
    
    def get_all_channels(self):
        """获取所有通道配置"""
        return self.channels


class BScanMatrix:
    """B显矩阵管理类"""
    
    def __init__(self, max_channels: int = 9, config_data=None):
        """
        初始化B显矩阵管理器
        
        Args:
            max_channels: 最大通道数
            config_data: 通道配置数据（可选）
        """
        self.max_channels = max_channels
        self.max_encoding = 0
        self.bscan_matrix = None
        
        # 使用通道管理器管理通道配置
        self.channel_manager = ChannelManager(config_data)
        
        # 初始化通道配置列表，通道号从1开始
        self.channel_configs = [None] * (max_channels + 1)  # 索引0未使用
        for channel_id in range(1, max_channels + 1):
            self.channel_configs[channel_id] = self.channel_manager.get_channel_config(channel_id)
    
    def init_bscan_matrix(self, max_encoding: int):
        """
        初始化B显数据矩阵
        
        Args:
            max_encoding: 最大编码值
        """
        self.max_encoding = max_encoding
        # 初始化B显矩阵，每个元素为BScanPoint对象
        self.bscan_matrix = [[BScanPoint() for _ in range(max_encoding)] for _ in range(self.max_channels)]
    
    def update_bscan_matrix(self, channel: int, encoding: int, depth: float, db_value: float, amplitude: float = 100.0):
        """
        更新B显数据矩阵，新编码值从右向左填充
        根据需求文档：矩阵最右一列为起始编码值，新编码值对应产生的深度值自右向左填充
        
        Args:
            channel: 通道号（1-9）
            encoding: 编码值
            depth: 深度值（mm）
            db_value: 原始超声信号强度（dB）
            amplitude: 信号幅度值（%），默认为100%
        """
        # 转换为0-based索引
        channel_idx = channel - 1
        if 0 <= channel_idx < self.max_channels and 0 <= encoding < self.max_encoding:
            # 计算实际填充位置，从右向左填充
            # 起始编码值（0）放在最右边，新编码值放在左边
            actual_col = self.max_encoding - 1 - encoding
            if 0 <= actual_col < self.max_encoding:
                # 更新BScanPoint对象，包含幅度值
                self.bscan_matrix[channel_idx][actual_col] = BScanPoint(depth, db_value, amplitude)
    
    def calculate_x_coordinate(self, current_x: float, channel: int, encoding_value: float) -> float:
        """
        计算X坐标，B显左上角为原点，x向右
        根据需求文档：X = X + b + Δx，其中b是拼孔值，Δx是编码值
        
        Args:
            current_x: 当前X坐标
            channel: 通道号（1-9）
            encoding_value: 编码值（Δx）
            
        Returns:
            计算得到的X坐标
        """
        if 1 <= channel <= self.max_channels and self.channel_configs[channel] is not None:
            offset = self.channel_configs[channel].offset  # 拼孔值b
            return current_x + offset + encoding_value  # X = X + b + Δx
        return current_x
    
    def calculate_y_coordinate(self, depth: float, y_max: float) -> float:
        """
        计算Y坐标，B显左上角为原点，y向下
        根据需求文档：Y=y/3=Ymax*d/3
        
        Args:
            depth: 深度值（mm）
            y_max: 最大Y坐标
            
        Returns:
            计算得到的Y坐标
        """
        # 深度值d对应Y坐标：Y = Ymax * d / 3
        # 由于y向下，深度越大，Y坐标越大
        return y_max * depth / 3
    
    def get_bscan_data(self, data_type: str = "depth") -> List[List[float]]:
        """
        获取B显数据矩阵
        
        Args:
            data_type: 数据类型，可选值："depth"（深度值）、"db"（原始dB值）、"amplitude"（幅度值）
            
        Returns:
            B显数据矩阵
        """
        if self.bscan_matrix is None:
            return None
        
        # 根据数据类型提取相应的值
        result = []
        for channel_data in self.bscan_matrix:
            channel_result = []
            for point in channel_data:
                if data_type == "depth":
                    value = point.depth if point.depth is not None else 0.0
                elif data_type == "db":
                    value = point.db_value if point.db_value is not None else 0.0
                elif data_type == "amplitude":
                    value = point.amplitude if point.amplitude is not None else 100.0
                else:
                    value = point.depth if point.depth is not None else 0.0
                channel_result.append(value)
            result.append(channel_result)
        return result
    
    def get_bscan_points(self) -> List[List['BScanPoint']]:
        """
        获取原始B显数据点矩阵
        
        Returns:
            BScanPoint对象矩阵
        """
        return self.bscan_matrix


class HyperdimensionalEncoder:
    """超维计算编码器"""
    
    def __init__(self, d=10000, seed=42):
        """
        初始化超维编码器
        Args:
            d: 超维向量维度
            seed: 随机种子
        """
        self.d = d
        self.seed = seed
        np.random.seed(seed)
        
        # 项目存储器：存储地址和数值超维向量
        self.item_memory = {
            'channels': {},      # 通道地址向量
            'codes': {},         # 编码值地址向量  
            'depths': {},        # 深度数值向量
            'amplitudes': {}     # 幅度数值向量
        }
        
        # 联想存储器：存储类别超向量
        self.associative_memory = {}
        
    def generate_binary_hypervector(self):
        """生成随机二元超维向量 (-1, 1)"""
        return np.random.choice([-1, 1], size=self.d)
    
    def generate_address_vector(self, key, memory_type):
        """生成地址超维向量并存储"""
        if key not in self.item_memory[memory_type]:
            self.item_memory[memory_type][key] = self.generate_binary_hypervector()
        return self.item_memory[memory_type][key]
    
    def continuous_mapping(self, min_val, max_val, num_levels, base_key):
        """
        连续映射生成数值超维向量
        Args:
            min_val: 最小值
            max_val: 最大值
            num_levels: 离散级别数
            base_key: 存储器键前缀
        """
        # 初始化存储器
        if base_key not in self.item_memory:
            self.item_memory[base_key] = {}
        
        # 生成最小值向量
        base_vector = self.generate_binary_hypervector()
        self.item_memory[base_key][min_val] = base_vector
        
        # 计算每次翻转的位数
        flip_bits = self.d // (2 * (num_levels - 1))
        
        # 生成连续映射的数值向量
        for i in range(1, num_levels):
            # 从前一个向量翻转固定位数
            prev_vector = self.item_memory[base_key][min_val + (i-1)]
            flip_indices = np.random.choice(self.d, flip_bits, replace=False)
            new_vector = prev_vector.copy()
            new_vector[flip_indices] *= -1  # 翻转位
            
            value = min_val + i
            self.item_memory[base_key][value] = new_vector
            
        return self.item_memory[base_key]
    
    def circular_shift_encode(self, depth_values: List[float], min_val: float, max_val: float, num_levels: int):
        """
        循环移位编码，用于表示多个点的深度值序列
        每个点的深度值对应一个超维向量，多个点的深度值可以通过循环移位编码表示为一个超维向量
        
        Args:
            depth_values: 深度值列表
            min_val: 最小深度值
            max_val: 最大深度值
            num_levels: 离散级别数
            
        Returns:
            循环移位编码后的超维向量
        """
        # 处理空列表情况
        if not depth_values:
            return self.generate_binary_hypervector()
        
        # 生成基础深度向量
        base_key = 'depths_circular'
        if base_key not in self.item_memory:
            self.item_memory[base_key] = {}
        
        # 使用连续映射生成深度向量
        self.continuous_mapping(min_val, max_val, num_levels, base_key)
        
        # 将深度值转换为整数索引
        depth_indices = []
        for depth in depth_values:
            # 将深度值映射到离散级别
            normalized_depth = (depth - min_val) / (max_val - min_val)
            index = int(normalized_depth * (num_levels - 1))
            index = max(0, min(index, num_levels - 1))
            depth_indices.append(index)
        
        # 获取第一个深度值对应的向量作为基础向量
        base_vector = self.item_memory[base_key][min_val + depth_indices[0]]
        
        # 对后续深度值进行循环移位
        result_vector = base_vector.copy()
        for i in range(1, len(depth_indices)):
            depth_vector = self.item_memory[base_key][min_val + depth_indices[i]]
            # 真正的循环移位操作：将结果向量循环移位i位，然后与当前深度向量绑定
            shifted_vector = np.roll(result_vector, i)
            result_vector = self.bind(shifted_vector, depth_vector)
        
        return result_vector
    
    def encode_damage_region(self, region_points: List[Tuple[int, int, float, float]], 
                           min_depth: float, max_depth: float, 
                           min_amplitude: float, max_amplitude: float, 
                           num_depth_levels: int = 301, num_amplitude_levels: int = 101):
        """
        编码完整的伤损区域
        一个伤损区域由多个点构成，每个点包含(channel, code, depth, amplitude)
        
        Args:
            region_points: 伤损区域的点列表，每个点格式为(channel, code, depth, amplitude)
            min_depth: 最小深度值
            max_depth: 最大深度值
            min_amplitude: 最小幅度值
            max_amplitude: 最大幅度值
            num_depth_levels: 深度离散级别数
            num_amplitude_levels: 幅度离散级别数
            
        Returns:
            伤损区域的超维编码向量
        """
        if not region_points:
            return self.generate_binary_hypervector()
        
        # 按通道和编码值排序，保留空间关系
        sorted_points = sorted(region_points, key=lambda x: (x[0], x[1]))
        
        # 为每个通道生成编码向量
        channel_vectors = defaultdict(list)
        for point in sorted_points:
            channel, code, depth, amplitude = point
            
            # 获取或生成通道向量
            channel_vec = self.generate_address_vector(channel, 'channels')
            
            # 获取或生成编码值向量
            code_vec = self.generate_address_vector(code, 'codes')
            
            # 确保深度和幅度向量已生成
            if 'depths' not in self.item_memory or not self.item_memory['depths']:
                self.continuous_mapping(min_depth, max_depth, num_depth_levels, 'depths')
            if 'amplitudes' not in self.item_memory or not self.item_memory['amplitudes']:
                self.continuous_mapping(min_amplitude, max_amplitude, num_amplitude_levels, 'amplitudes')
            
            # 获取深度和幅度向量
            depth_index = int((depth - min_depth) / (max_depth - min_depth) * (num_depth_levels - 1))
            depth_index = max(0, min(depth_index, num_depth_levels - 1))
            depth_vec = self.item_memory['depths'].get(min_depth + depth_index, self.generate_binary_hypervector())
            
            amplitude_index = int((amplitude - min_amplitude) / (max_amplitude - min_amplitude) * (num_amplitude_levels - 1))
            amplitude_index = max(0, min(amplitude_index, num_amplitude_levels - 1))
            amplitude_vec = self.item_memory['amplitudes'].get(min_amplitude + amplitude_index, self.generate_binary_hypervector())
            
            # 编码单个点
            point_vec = self.bundle([
                self.bind(channel_vec, depth_vec),
                self.bind(code_vec, depth_vec),
                self.bind(channel_vec, amplitude_vec),
                self.bind(code_vec, amplitude_vec)
            ])
            
            channel_vectors[channel].append(point_vec)
        
        # 对每个通道的点向量进行循环移位编码
        channel_circular_vectors = []
        for channel, vectors in channel_vectors.items():
            # 计算通道权重（基于点数量）
            weight = len(vectors) / len(sorted_points)
            
            # 对通道内的点向量进行循环移位
            if len(vectors) == 1:
                # 单个点直接使用
                circular_vec = vectors[0]
            else:
                # 多个点使用循环移位
                circular_vec = vectors[0].copy()
                for i in range(1, len(vectors)):
                    shifted_vec = np.roll(circular_vec, i)
                    circular_vec = self.bind(shifted_vec, vectors[i])
            
            # 应用权重（通过重复捆绑）
            weighted_vec = circular_vec
            for _ in range(int(weight * 10)):
                weighted_vec = self.bundle([weighted_vec, circular_vec])
            
            channel_circular_vectors.append(weighted_vec)
        
        # 捆绑所有通道的向量，生成最终的伤损区域向量
        return self.bundle(channel_circular_vectors)
    
    def bind(self, address_vector, value_vector):
        """绑定运算（乘法）"""
        return address_vector * value_vector
    
    def bundle(self, vectors, normalize=True):
        """捆绑运算（加法）"""
        if not vectors:
            return np.zeros(self.d)
        
        bundled = np.sum(vectors, axis=0)
        
        if normalize:
            # 二值化处理：多数表决
            bundled = np.where(bundled > 0, 1, -1)
            
        return bundled
    
    def encode_value_with_segments(self, value: float, value_range: Tuple[float, float], num_segments: int = 10) -> np.ndarray:
        """
        分段编码，将连续值映射到离散段
        
        Args:
            value: 要编码的值
            value_range: 值的范围 (min_val, max_val)
            num_segments: 分段数
            
        Returns:
            编码后的超维向量
        """
        # 归一化值到[0, 1]范围
        normalized_value = (value - value_range[0]) / (value_range[1] - value_range[0])
        normalized_value = max(0.0, min(1.0, normalized_value))
        
        # 计算段索引
        segment_index = int(normalized_value * num_segments)
        segment_index = min(segment_index, num_segments - 1)
        
        # 生成段键
        segment_key = f"segment_{value_range[0]}_{value_range[1]}_{num_segments}_{segment_index}"
        
        # 生成或获取段向量
        if segment_key not in self.item_memory:
            self.item_memory[segment_key] = self.generate_binary_hypervector()
        
        return self.item_memory[segment_key]
    
    def encode_feature_interaction(self, feature1: np.ndarray, feature2: np.ndarray) -> np.ndarray:
        """
        特征交互编码，通过绑定运算实现特征之间的交互
        
        Args:
            feature1: 第一个特征向量
            feature2: 第二个特征向量
            
        Returns:
            交互编码后的向量
        """
        return self.bind(feature1, feature2)
    
    def encode_sample(self, channel, code, depth, amplitude):
        """
        编码单个B显数据点
        Args:
            channel: 通道号
            code: 编码值
            depth: 深度值
            amplitude: 幅度值
        Returns:
            样本超维向量
        """
        # 生成/获取地址向量
        h_channel = self.generate_address_vector(channel, 'channels')
        h_code = self.generate_address_vector(code, 'codes')
        
        # 获取数值向量（假设已通过continuous_mapping生成）
        v_depth = self.item_memory['depths'].get(depth, self.generate_binary_hypervector())
        v_amplitude = self.item_memory['amplitudes'].get(amplitude, self.generate_binary_hypervector())
        
        # 记录型编码：绑定地址和数值，然后捆绑
        bound_vectors = [
            self.bind(h_channel, v_depth),
            self.bind(h_code, v_depth),
            self.bind(h_channel, v_amplitude),
            self.bind(h_code, v_amplitude)
        ]
        
        # 捆绑所有绑定对
        sample_vector = self.bundle(bound_vectors, normalize=True)
        
        return sample_vector
    
    def encode_sample_enhanced(self, channel, code, depth, amplitude, depth_change=0.0, amplitude_change=0.0):
        """
        增强的样本编码方法，包含更多特征和特征交互
        
        Args:
            channel: 通道号
            code: 编码值
            depth: 深度值
            amplitude: 幅度值
            depth_change: 深度变化率
            amplitude_change: 幅度变化率
            
        Returns:
            增强编码后的样本超维向量
        """
        # 生成/获取地址向量
        h_channel = self.generate_address_vector(channel, 'channels')
        h_code = self.generate_address_vector(code, 'codes')
        
        # 获取数值向量
        v_depth = self.item_memory['depths'].get(depth, self.generate_binary_hypervector())
        v_amplitude = self.item_memory['amplitudes'].get(amplitude, self.generate_binary_hypervector())
        
        # 编码变化率（使用分段编码）
        v_depth_change = self.encode_value_with_segments(depth_change, (-50.0, 50.0), num_segments=10)
        v_amplitude_change = self.encode_value_with_segments(amplitude_change, (-50.0, 50.0), num_segments=10)
        
        # 基础绑定对
        bound_vectors = [
            self.bind(h_channel, v_depth),
            self.bind(h_code, v_depth),
            self.bind(h_channel, v_amplitude),
            self.bind(h_code, v_amplitude)
        ]
        
        # 添加变化率编码
        bound_vectors.append(v_depth_change)
        bound_vectors.append(v_amplitude_change)
        
        # 添加特征交互
        interaction1 = self.encode_feature_interaction(v_depth, v_amplitude)
        interaction2 = self.encode_feature_interaction(h_channel, h_code)
        bound_vectors.append(interaction1)
        bound_vectors.append(interaction2)
        
        # 捆绑所有向量
        sample_vector = self.bundle(bound_vectors, normalize=True)
        
        return sample_vector
    
    def hamming_similarity(self, vec1, vec2):
        """计算汉明相似度（归一化）"""
        # 对于二元向量(-1,1)，汉明相似度 = (1 - 汉明距离/d)
        matches = np.sum(vec1 == vec2)
        return matches / self.d
    
    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度（适用于实数向量）"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2 + 1e-10)


class HDCClassifier:
    """超维计算分类器"""
    
    def __init__(self, encoder):
        """
        初始化HDC分类器
        Args:
            encoder: HyperdimensionalEncoder实例
        """
        self.encoder = encoder
        self.class_vectors = {}  # 类别超向量
        self.class_labels = {}   # 类别标签映射
        
        # 深度和幅度的范围设置
        self.min_depth = 0.0
        self.max_depth = 300.0
        self.min_amplitude = 0.0
        self.max_amplitude = 100.0
    
    def determine_flaw_type(self, channel: int, depth: float, amplitude: float) -> str:
        """
        根据通道和深度判断伤损类型
        根据需求文档：伤损1-6是由直70°通道和斜70°通道所能发现的伤损，伤损7-11是由37°和0°通道所能发现的伤损
        
        Args:
            channel: 通道号
            depth: 深度值（mm）
            amplitude: 幅度值（%）
            
        Returns:
            伤损类型
        """
        # 核伤类：通道1-6（直70°和斜70°通道）
        if channel in [1, 2, 3, 4, 5, 6]:
            if depth > 100 and amplitude > 70:
                return "核伤类"
        
        # 水平裂纹类：通道7-8（37°通道）
        elif channel in [7, 8]:
            if depth > 120 and amplitude > 60:
                return "水平裂纹类"
        
        # 轨底裂纹类：通道9（0°通道）
        elif channel == 9:
            if depth > 150 and amplitude > 50:
                return "轨底裂纹类"
        
        # 螺孔裂纹类：特定深度范围
        if 80 < depth < 100 and amplitude > 50:
            return "螺孔裂纹类"
        
        # 其他类
        return "其他类"
    
    def determine_normal_structure(self, channel: int, depth: float, amplitude: float) -> str:
        """
        判断正常结构类型
        根据需求文档：正常结构包括钢轨接头、螺孔、导线孔、焊缝、鱼鳞伤、轨底锈蚀、轨底焊筋、道岔等
        
        Args:
            channel: 通道号
            depth: 深度值（mm）
            amplitude: 幅度值（%）
            
        Returns:
            正常结构类型
        """
        # 接头：特定深度范围，低幅度
        if 60 < depth < 80 and amplitude < 40:
            return "接头"
        
        # 螺孔：深度约90mm
        elif 85 < depth < 95 and amplitude < 50:
            return "螺孔"
        
        # 导线孔：深度约110mm
        elif 105 < depth < 115 and amplitude < 45:
            return "导线孔"
        
        # 焊缝：深度约70mm
        elif 65 < depth < 75 and amplitude < 50:
            return "焊缝"
        
        # 鱼鳞伤：表面附近，中等幅度
        elif depth < 50 and 30 < amplitude < 60:
            return "鱼鳞伤"
        
        # 轨底锈蚀：大深度，低幅度
        elif depth > 180 and amplitude < 30:
            return "轨底锈蚀"
        
        # 轨底焊筋：特定深度范围
        elif 130 < depth < 150 and amplitude < 40:
            return "轨底焊筋"
        
        # 道岔：特定深度范围
        elif 100 < depth < 130 and amplitude < 50:
            return "道岔"
        
        # 其他
        return "其他"
    
    def train(self, samples, labels):
        """
        训练HDC分类器
        Args:
            samples: 样本列表，每个样本为(channel, code, depth, amplitude)
            labels: 对应标签列表
        """
        # 按类别分组样本
        class_groups = defaultdict(list)
        for sample, label in zip(samples, labels):
            class_groups[label].append(sample)
        
        # 为每个类别生成类别超向量
        for label, sample_list in class_groups.items():
            sample_vectors = []
            
            for sample in sample_list:
                channel, code, depth, amplitude = sample
                # 编码样本
                sample_vector = self.encoder.encode_sample(channel, code, depth, amplitude)
                sample_vectors.append(sample_vector)
            
            # 捆绑同类所有样本向量
            class_vector = self.encoder.bundle(sample_vectors, normalize=True)
            
            # 存储类别超向量
            self.class_vectors[label] = class_vector
            self.class_labels[label] = label
            
            # 存入联想存储器
            self.encoder.associative_memory[label] = class_vector
    
    def train_with_regions(self, damage_regions, normal_regions):
        """
        使用伤损区域和正常区域进行训练
        Args:
            damage_regions: 伤损区域列表，每个元素为(region_points, label)
            normal_regions: 正常区域列表，每个元素为(region_points, label)
        """
        # 合并所有区域
        all_regions = damage_regions + normal_regions
        
        # 按类别分组区域
        class_groups = defaultdict(list)
        for region_points, label in all_regions:
            class_groups[label].append(region_points)
        
        # 为每个类别生成类别超向量
        for label, region_list in class_groups.items():
            region_vectors = []
            
            for region_points in region_list:
                # 编码整个伤损区域
                region_vector = self.encoder.encode_damage_region(
                    region_points, 
                    self.min_depth, self.max_depth, 
                    self.min_amplitude, self.max_amplitude
                )
                region_vectors.append(region_vector)
            
            # 捆绑同类所有区域向量
            class_vector = self.encoder.bundle(region_vectors, normalize=True)
            
            # 存储类别超向量
            self.class_vectors[label] = class_vector
            self.class_labels[label] = label
            
            # 存入联想存储器
            self.encoder.associative_memory[label] = class_vector
    
    def predict(self, sample):
        """
        预测样本类别
        Args:
            sample: 待预测样本 (channel, code, depth, amplitude)
        Returns:
            预测标签，相似度字典
        """
        # 编码查询样本
        channel, code, depth, amplitude = sample
        query_vector = self.encoder.encode_sample(channel, code, depth, amplitude)
        
        # 计算与每个类别超向量的相似度
        similarities = {}
        for label, class_vector in self.class_vectors.items():
            similarity = self.encoder.hamming_similarity(query_vector, class_vector)
            similarities[label] = similarity
        
        # 选择相似度最高的类别
        predicted_label = max(similarities.items(), key=lambda x: x[1])[0]
        
        return predicted_label, similarities
    
    def predict_region(self, region_points):
        """
        预测伤损区域的类别
        Args:
            region_points: 伤损区域的点列表
        Returns:
            预测标签，相似度字典
        """
        # 编码查询区域
        query_vector = self.encoder.encode_damage_region(
            region_points, 
            self.min_depth, self.max_depth, 
            self.min_amplitude, self.max_amplitude
        )
        
        # 计算与每个类别超向量的相似度
        similarities = {}
        for label, class_vector in self.class_vectors.items():
            similarity = self.encoder.hamming_similarity(query_vector, class_vector)
            similarities[label] = similarity
        
        # 选择相似度最高的类别
        predicted_label = max(similarities.items(), key=lambda x: x[1])[0]
        
        return predicted_label, similarities
    
    def scan_with_sliding_window(self, bscan_samples, window_size=5, step=2, min_similarity=0.7):
        """
        使用滑动窗口扫描B显数据，检测伤损区域
        Args:
            bscan_samples: B显样本列表
            window_size: 滑动窗口大小（点数）
            step: 滑动步长
            min_similarity: 最小相似度阈值
        Returns:
            检测到的伤损区域列表，每个元素为(region_points, predicted_label, similarity)
        """
        # 按通道分组
        channel_samples = defaultdict(list)
        for sample in bscan_samples:
            channel, code, depth, amplitude, db_value = sample
            channel_samples[channel].append((channel, code, depth, amplitude))
        
        detected_regions = []
        
        # 记录所有相似性得分
        self.similarity_scores = []
        
        # 对每个通道进行滑动窗口扫描
        for channel, samples in channel_samples.items():
            # 按编码值排序
            sorted_samples = sorted(samples, key=lambda x: x[1])
            
            # 滑动窗口扫描
            for i in range(0, len(sorted_samples) - window_size + 1, step):
                # 提取窗口内的点
                window_points = sorted_samples[i:i+window_size]
                
                # 预测窗口内的区域
                predicted_label, similarities = self.predict_region(window_points)
                max_similarity = max(similarities.values())
                
                # 记录相似性得分
                self.similarity_scores.append(max_similarity)
                
                # 如果相似度超过阈值，保存检测结果
                if max_similarity >= min_similarity:
                    detected_regions.append((window_points, predicted_label, max_similarity))
        
        return detected_regions
    
    def retrain(self, samples, labels, iterations=10, learning_rate=0.1):
        """
        重训练优化（改进版蒙版重训练）
        Args:
            samples: 训练样本
            labels: 训练标签
            iterations: 重训练迭代次数
            learning_rate: 学习率
        """
        for iteration in range(iterations):
            misclassified = []
            
            # 找出分类错误的样本
            for i, (sample, true_label) in enumerate(zip(samples, labels)):
                pred_label, _ = self.predict(sample)
                if pred_label != true_label:
                    misclassified.append((i, sample, true_label, pred_label))
            
            if not misclassified:
                print(f"迭代 {iteration+1}: 所有样本分类正确")
                break
            
            # 调整类别超向量（使用实数向量和增量学习）
            for idx, sample, true_label, wrong_label in misclassified:
                # 编码错误样本
                channel, code, depth, amplitude = sample
                sample_vector = self.encoder.encode_sample(channel, code, depth, amplitude)
                
                # 从错误类别中移除，添加到正确类别
                # 使用实数向量，不立即归一化
                if wrong_label in self.class_vectors:
                    self.class_vectors[wrong_label] = self.class_vectors[wrong_label] - learning_rate * sample_vector
                
                if true_label in self.class_vectors:
                    self.class_vectors[true_label] = self.class_vectors[true_label] + learning_rate * sample_vector
            
            # 迭代结束后统一归一化
            for label in self.class_vectors:
                norm = np.linalg.norm(self.class_vectors[label])
                if norm > 0:
                    self.class_vectors[label] = self.class_vectors[label] / norm
                # 二值化
                self.class_vectors[label] = np.where(self.class_vectors[label] > 0, 1, -1)
            
            print(f"迭代 {iteration+1}: 错误样本数 {len(misclassified)}")
    
    def set_depth_range(self, min_depth, max_depth):
        """
        设置深度值范围
        Args:
            min_depth: 最小深度值
            max_depth: 最大深度值
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def set_amplitude_range(self, min_amplitude, max_amplitude):
        """
        设置幅度值范围
        Args:
            min_amplitude: 最小幅度值
            max_amplitude: 最大幅度值
        """
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude


class BScanDataProcessor:
    """B显数据处理器"""
    
    def __init__(self, encoder, config_data=None):
        self.encoder = encoder
        self.bscan_matrix = BScanMatrix(config_data=config_data)
        self.damage_types = {
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
            11: "螺孔裂纹",
            12: "其他伤损"
        }
        
        self.normal_types = {
            1: "接头",
            2: "螺孔",
            3: "导线孔",
            4: "焊缝",
            5: "鱼鳞伤",
            6: "轨底锈蚀",
            7: "轨底焊筋",
            8: "道岔",
            9: "其他"
        }
    
    def load_json_data(self, json_file):
        """从JSON文件加载B显数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        labels = []
        encoding_count = 0  # 初始化编码计数
        
        # 解析配置数据（第一个元素通常是配置）
        config_data = None
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'E' in data[0]:
                config_data = data[0]
                encoding_count = config_data['E']  # 从配置获取初始编码计数
                data = data[1:]  # 剩下的是B显数据
                
                # 更新通道配置
                self.bscan_matrix.channel_manager.parse_config(config_data)
                # 更新通道配置列表
                for channel_id in range(1, self.bscan_matrix.max_channels + 1):
                    self.bscan_matrix.channel_configs[channel_id] = self.bscan_matrix.channel_manager.get_channel_config(channel_id)
        
        # 解析B显数据
        for item in data:
            if isinstance(item, dict):
                # 检查是否有E字段，更新编码计数
                if 'E' in item:
                    encoding_count = item['E']
                
                for key, value in item.items():
                    # 解析JSON格式: {"B1":[1,125,80]}
                    if isinstance(value, list) and len(value) >= 3:
                        channel = int(value[0])  # 确保通道号是整数
                        acoustic_range = value[1]  # 原始声程
                        amplitude = value[2]
                        code = encoding_count  # 使用当前编码计数作为编码值
                        
                        # 获取通道配置
                        channel_config = self.bscan_matrix.channel_configs[channel]
                        if channel_config:
                            # 进行角度矫正，计算实际深度
                            depth = calculate_actual_depth(
                                acoustic_range, 
                                channel_config.horizontal_angle, 
                                channel_config.pitch_angle
                            )
                            # 计算编码偏差
                            encoding_offset = calculate_encoding_offset(
                                acoustic_range, 
                                channel_config.horizontal_angle, 
                                channel_config.pitch_angle
                            )
                        else:
                            # 如果没有通道配置，使用原始声程作为深度
                            depth = acoustic_range
                            encoding_offset = 0
                        
                        # 计算dB值（假设dB值与幅度值线性相关）
                        # 确保amplitude为正数，避免log10计算错误
                        normalized_amplitude = amplitude / 100
                        if normalized_amplitude > 0:
                            db_value = 20 * np.log10(normalized_amplitude) + 50
                        else:
                            # 对于0或负数幅度，设置一个默认的dB值（例如最小值）
                            db_value = -np.inf
                        
                        samples.append((channel, code, depth, amplitude, db_value))
                        
                        # 自动生成标签
                        label = self._determine_label(channel, depth, amplitude)
                        labels.append(label)
        
        return samples, labels, config_data
    
    def load_annotation_data(self, annotation_file):
        """从标注JSON文件加载标注数据"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        damage_regions = []
        normal_regions = []
        
        # 解析标注数据
        annotations = annotation_data.get('annotations', [])
        
        for item in annotations:
            if isinstance(item, dict):
                # 标注格式：type（类型）、points（点列表）、subtype（子类型）
                region_type = item.get('type', '')
                points = item.get('points', [])
                subtype = item.get('subtype', {})
                
                # 获取标签名称
                label_name = subtype.get('name', '未知类型')
                
                # 转换点格式：从标注坐标转换为B显点格式
                bscan_points = []
                for point in points:
                    if isinstance(point, dict):
                        e = point.get('e', 0)
                        channel = point.get('channel', 0)
                        depth = point.get('depth', 0.0)
                        amplitude = point.get('amplitude', 0.0)
                        bscan_points.append((channel, e, depth, amplitude))
                
                # 根据类型区分伤损和正常区域
                if region_type == '伤损':
                    damage_regions.append((bscan_points, label_name))
                elif region_type == '结构':
                    # 结构作为正常区域处理
                    normal_regions.append((bscan_points, label_name))
        
        return damage_regions, normal_regions
    
    def extract_damage_regions_from_bscan(self, bscan_samples, min_points=2):
        """从B显数据中提取潜在的伤损区域（用于无标注数据的自动提取）"""
        # 按通道分组
        channel_samples = defaultdict(list)
        for sample in bscan_samples:
            channel, code, depth, amplitude, db_value = sample
            channel_samples[channel].append(sample)
        
        damage_regions = []
        
        # 对每个通道的样本进行处理
        for channel, samples in channel_samples.items():
            # 按编码值排序
            sorted_samples = sorted(samples, key=lambda x: x[1])
            
            current_region = []
            previous_code = None
            
            for sample in sorted_samples:
                channel, code, depth, amplitude, db_value = sample
                
                # 判断是否为潜在伤损点（幅度大于阈值）
                if amplitude > 60:  # 幅度阈值
                    if previous_code is None or code <= previous_code + 3:  # 编码值连续或间隔小于等于3
                        # 继续当前区域
                        current_region.append((channel, code, depth, amplitude))
                    else:
                        # 保存当前区域并开始新区域
                        if len(current_region) >= min_points:
                            damage_regions.append(current_region)
                        current_region = [(channel, code, depth, amplitude)]
                    previous_code = code
                else:
                    # 结束当前区域
                    if len(current_region) >= min_points:
                        damage_regions.append(current_region)
                    current_region = []
                    previous_code = None
            
            # 保存最后一个区域
            if len(current_region) >= min_points:
                damage_regions.append(current_region)
        
        return damage_regions
    
    def _determine_label(self, channel, depth, amplitude):
        """根据通道和深度确定标签"""
        # 首先判断是否为伤损
        flaw_type = self._determine_flaw_type(channel, depth, amplitude)
        if flaw_type != "其他类":
            return flaw_type
        
        # 如果不是伤损，判断是否为正常结构
        normal_type = self._determine_normal_structure(channel, depth, amplitude)
        return normal_type
    
    def _determine_flaw_type(self, channel, depth, amplitude):
        """判断伤损类型"""
        # 核伤类：通道1-6
        if channel in [1, 2, 3, 4, 5, 6]:
            if depth > 100 and amplitude > 70:
                return "核伤类"
        
        # 水平裂纹类：通道7-8
        elif channel in [7, 8]:
            if depth > 120 and amplitude > 60:
                return "水平裂纹类"
        
        # 轨底裂纹类：通道9
        elif channel == 9:
            if depth > 150 and amplitude > 50:
                return "轨底裂纹类"
        
        # 螺孔裂纹类
        if 80 < depth < 100 and amplitude > 50:
            return "螺孔裂纹类"
        
        return "其他类"
    
    def _determine_normal_structure(self, channel, depth, amplitude):
        """判断正常结构类型"""
        if 60 < depth < 80 and amplitude < 40:
            return "接头"
        elif 85 < depth < 95 and amplitude < 50:
            return "螺孔"
        elif 105 < depth < 115 and amplitude < 45:
            return "导线孔"
        elif 65 < depth < 75 and amplitude < 50:
            return "焊缝"
        elif depth < 50 and 30 < amplitude < 60:
            return "鱼鳞伤"
        elif depth > 180 and amplitude < 30:
            return "轨底锈蚀"
        elif 130 < depth < 150 and amplitude < 40:
            return "轨底焊筋"
        elif 100 < depth < 130 and amplitude < 50:
            return "道岔"
        return "其他"
    
    def calculate_changes(self, samples):
        """计算数据变化率"""
        samples_with_changes = []
        
        # 按通道和编码值排序
        sorted_samples = sorted(samples, key=lambda x: (x[0], x[1]))
        
        previous_points = {}
        for sample in sorted_samples:
            channel, code, depth, amplitude, db_value = sample
            
            # 计算变化率
            depth_change = 0.0
            amplitude_change = 0.0
            
            prev_key = (channel, code - 1)
            if prev_key in previous_points:
                prev_point = previous_points[prev_key]
                depth_change = depth - prev_point[2]
                amplitude_change = amplitude - prev_point[3]
            
            samples_with_changes.append((channel, code, depth, amplitude, db_value, depth_change, amplitude_change))
            previous_points[(channel, code)] = sample
        
        return samples_with_changes


# 使用示例
def main():
    # 1. 初始化编码器
    encoder = HyperdimensionalEncoder(d=10000, seed=42)
    
    # 2. 使用连续映射生成深度和幅度的数值向量
    encoder.continuous_mapping(min_val=0, max_val=300, num_levels=301, base_key='depths')
    encoder.continuous_mapping(min_val=0, max_val=100, num_levels=101, base_key='amplitudes')
    
    # 3. 准备数据
    processor = BScanDataProcessor(encoder)
    samples, labels = processor.create_training_data(num_samples=500)
    
    # 计算变化率
    samples_with_changes = processor.calculate_changes(samples)
    
    # 转换为训练格式（不包含db_value和变化率，保持原有格式）
    training_samples = [(s[0], s[1], s[2], s[3]) for s in samples_with_changes]
    
    # 4. 训练分类器
    classifier = HDCClassifier(encoder)
    classifier.train(training_samples, labels)
    
    # 5. 测试
    test_sample = (1, 0, 125, 80)  # 通道1, 编码值0, 深度125mm, 幅度80%
    predicted_label, similarities = classifier.predict(test_sample)
    
    print(f"测试样本预测结果: {predicted_label}")
    print(f"各类别相似度: {similarities}")
    
    # 6. 重训练优化
    print("\n开始重训练优化...")
    classifier.retrain(training_samples, labels, iterations=5, learning_rate=0.1)
    
    # 7. 重训练后重新测试
    predicted_label, similarities = classifier.predict(test_sample)
    print(f"\n重训练后预测结果: {predicted_label}")
    print(f"各类别相似度: {similarities}")
    
    # 8. 测试循环移位编码
    print("\n测试循环移位编码...")
    depth_values = [100, 110, 120, 130]
    circular_vector = encoder.circular_shift_encode(depth_values, min_val=0, max_val=300, num_levels=301)
    print(f"循环移位编码向量维度: {circular_vector.shape}")
    print(f"向量元素唯一值: {np.unique(circular_vector)}")
    
    # 9. 测试增强样本编码
    print("\n测试增强样本编码...")
    enhanced_vector = encoder.encode_sample_enhanced(
        channel=1, code=0, depth=125, amplitude=80,
        depth_change=5.0, amplitude_change=10.0
    )
    print(f"增强编码向量维度: {enhanced_vector.shape}")
    print(f"向量元素唯一值: {np.unique(enhanced_vector)}")


if __name__ == "__main__":
    main()
