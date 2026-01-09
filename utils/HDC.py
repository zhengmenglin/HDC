import numpy as np
import json
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional


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


class BScanMatrix:
    """B显矩阵管理类"""
    
    def __init__(self, max_channels: int = 9):
        """
        初始化B显矩阵管理器
        
        Args:
            max_channels: 最大通道数
        """
        self.max_channels = max_channels
        self.max_encoding = 0
        self.bscan_matrix = None
        
        # 初始化9个通道的配置（根据需求文档）
        # 通道配置：(俯仰角度, 水平角度, 拼孔值)
        self.channel_configs = [
            ChannelConfig(70, 20, 10),   # 通道1：俯仰70°, 水平角度α(默认20°), 拼孔值10
            ChannelConfig(70, 20, 60),   # 通道2：俯仰70°, 水平角度α(默认20°), 拼孔值60
            ChannelConfig(70, 20, 20),   # 通道3：俯仰70°, 水平角度α(默认20°), 拼孔值20
            ChannelConfig(70, 20, 50),   # 通道4：俯仰70°, 水平角度α(默认20°), 拼孔值50
            ChannelConfig(70, 0, 30),    # 通道5：俯仰70°, 水平角度0°, 拼孔值30
            ChannelConfig(70, 0, 40),    # 通道6：俯仰70°, 水平角度0°, 拼孔值40
            ChannelConfig(37, 0, 0),     # 通道7：俯仰37°, 水平角度0°, 拼孔值0
            ChannelConfig(37, 0, 70),     # 通道8：俯仰37°, 水平角度0°, 拼孔值70
            ChannelConfig(0, 0, 80)       # 通道9：俯仰0°, 水平角度0°, 拼孔值80
        ]
    
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
            channel: 通道号
            encoding: 编码值
            depth: 深度值（mm）
            db_value: 原始超声信号强度（dB）
            amplitude: 信号幅度值（%），默认为100%
        """
        if 0 <= channel < self.max_channels and 0 <= encoding < self.max_encoding:
            # 计算实际填充位置，从右向左填充
            # 起始编码值（0）放在最右边，新编码值放在左边
            actual_col = self.max_encoding - 1 - encoding
            if 0 <= actual_col < self.max_encoding:
                # 更新BScanPoint对象，包含幅度值
                self.bscan_matrix[channel][actual_col] = BScanPoint(depth, db_value, amplitude)
    
    def calculate_x_coordinate(self, current_x: float, channel: int, encoding_value: float) -> float:
        """
        计算X坐标，B显左上角为原点，x向右
        根据需求文档：X = X + b + Δx，其中b是拼孔值，Δx是编码值
        
        Args:
            current_x: 当前X坐标
            channel: 通道号
            encoding_value: 编码值（Δx）
            
        Returns:
            计算得到的X坐标
        """
        if 0 <= channel < self.max_channels:
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
            # 循环移位操作：将结果向量与深度向量进行异或操作
            result_vector = np.where(result_vector != depth_vector, -1, 1)
        
        return result_vector
    
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


class BScanDataProcessor:
    """B显数据处理器"""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.bscan_matrix = BScanMatrix()
        
    def load_json_data(self, json_file):
        """从JSON文件加载B显数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        labels = []
        
        for key, value in data.items():
            # 解析JSON格式: {"B1":[1,125,80]}
            if isinstance(value, list) and len(value) >= 3:
                channel = value[0]
                depth = value[1]
                amplitude = value[2]
                code = int(key[1:])  # 从B1、B2等中提取编码值
                
                # 计算dB值（假设dB值与幅度值线性相关）
                db_value = 20 * np.log10(amplitude / 100) + 50
                
                samples.append((channel, code, depth, amplitude, db_value))
                
                # 自动生成标签
                label = self._determine_label(channel, depth, amplitude)
                labels.append(label)
        
        return samples, labels
    
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
    
    def create_training_data(self, num_samples=1000):
        """生成模拟训练数据"""
        samples = []
        labels = []
        
        # 正常结构数据
        for _ in range(num_samples // 2):
            channel = random.randint(1, 9)
            code = random.randint(0, 100)
            depth = random.randint(50, 150)
            amplitude = random.randint(10, 50)
            db_value = 20 * np.log10(amplitude / 100) + 50
            samples.append((channel, code, depth, amplitude, db_value))
            labels.append(self._determine_label(channel, depth, amplitude))
        
        # 伤损数据
        for _ in range(num_samples // 2):
            channel = random.randint(1, 9)
            code = random.randint(0, 100)
            depth = random.randint(100, 200)
            amplitude = random.randint(60, 100)
            db_value = 20 * np.log10(amplitude / 100) + 50
            samples.append((channel, code, depth, amplitude, db_value))
            labels.append(self._determine_label(channel, depth, amplitude))
        
        return samples, labels
    
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
