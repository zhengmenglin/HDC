import torch
import torchhd
from torchhd.models import NeuralHD
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


class TorchHDEncoder:
    """基于torchhd的超维计算编码器"""
    
    def __init__(self, d=10000, n_features=4, seed=42, device="cpu"):
        """
        初始化超维编码器
        Args:
            d: 超维向量维度
            n_features: 特征数量（通道、编码值、深度、幅度）
            seed: 随机种子
            device: 设备（cpu或cuda）
        """
        self.d = d
        self.n_features = n_features
        self.seed = seed
        self.device = device
        
        # 固定随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 项目存储器：存储地址和数值超维向量
        self.item_memory = {
            'channels': {},      # 通道地址向量
            'codes': {},         # 编码值地址向量  
            'depths': {},        # 深度数值向量
            'amplitudes': {}     # 幅度数值向量
        }
        
        # 生成随机基向量用于连续映射
        self.base_vectors = {
            'depth': torchhd.random_hv(d, device=device),
            'amplitude': torchhd.random_hv(d, device=device)
        }
    
    def generate_hypervector(self):
        """生成随机超维向量"""
        return torchhd.random_hv(self.d, device=self.device)
    
    def generate_address_vector(self, key, memory_type):
        """生成地址超维向量并存储"""
        if key not in self.item_memory[memory_type]:
            self.item_memory[memory_type][key] = self.generate_hypervector()
        return self.item_memory[memory_type][key]
    
    def continuous_mapping(self, value, min_val, max_val, base_key):
        """
        连续映射生成数值超维向量
        基于论文《X超维计算概念,应用及研究进展》中的连续映射方法
        
        Args:
            value: 要映射的值
            min_val: 最小值
            max_val: 最大值
            base_key: 存储器键类型（'depth'或'amplitude'）
            
        Returns:
            数值超维向量
        """
        # 归一化值到[0, 1]范围
        normalized_value = (value - min_val) / (max_val - min_val)
        normalized_value = max(0.0, min(1.0, normalized_value))
        
        # 使用torchhd的连续映射功能
        # 基于基向量，通过插值生成连续变化的超向量
        # 这里使用旋转映射实现连续变化
        base_vector = self.base_vectors[base_key]
        
        # 生成噪声向量，模拟连续映射的细微变化
        noise = torch.randn_like(base_vector) * 0.1
        
        # 使用线性组合实现连续映射
        mapped_vector = base_vector + normalized_value * noise
        
        # 归一化到单位长度
        mapped_vector = mapped_vector / torch.norm(mapped_vector)
        
        return mapped_vector
    
    def circular_shift_encode(self, depth_values, min_val, max_val):
        """
        循环移位编码，用于表示多个点的深度值序列
        每个点的深度值对应一个超维向量，多个点的深度值可以通过循环移位编码表示为一个超维向量
        
        Args:
            depth_values: 深度值列表
            min_val: 最小深度值
            max_val: 最大深度值
            
        Returns:
            循环移位编码后的超维向量
        """
        # 获取第一个深度值对应的向量作为基础向量
        base_vector = self.continuous_mapping(depth_values[0], min_val, max_val, 'depth')
        
        # 对后续深度值进行循环移位
        result_vector = base_vector.clone()
        for i in range(1, len(depth_values)):
            depth_vector = self.continuous_mapping(depth_values[i], min_val, max_val, 'depth')
            # 循环移位操作：将结果向量与深度向量进行异或操作
            result_vector = torch.where(result_vector != depth_vector, -1.0, 1.0)
        
        return result_vector
    
    def encode_sample(self, channel, code, depth, amplitude):
        """
        编码单个B显数据点
        
        Args:
            channel: 通道号
            code: 编码值
            depth: 深度值
            amplitude: 幅度值
            
        Returns:
            样本超维向量 (特征向量)
        """
        # 使用连续映射生成数值向量
        depth_vector = self.continuous_mapping(depth, 0, 300, 'depth')
        amplitude_vector = self.continuous_mapping(amplitude, 0, 100, 'amplitude')
        
        # 生成地址向量
        channel_vector = self.generate_address_vector(channel, 'channels')
        code_vector = self.generate_address_vector(code, 'codes')
        
        # 特征融合：将所有特征向量连接成一个长向量
        # NeuralHD期望特征向量作为输入，而非单个超向量
        # 这里将4个特征向量（每个d维）连接成一个4d维的特征向量
        feature_vector = torch.cat([
            channel_vector,
            code_vector,
            depth_vector,
            amplitude_vector
        ])
        
        return feature_vector
    
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
            增强编码后的样本特征向量
        """
        # 使用连续映射生成数值向量
        depth_vector = self.continuous_mapping(depth, 0, 300, 'depth')
        amplitude_vector = self.continuous_mapping(amplitude, 0, 100, 'amplitude')
        depth_change_vector = self.continuous_mapping(depth_change, -50, 50, 'depth')
        amplitude_change_vector = self.continuous_mapping(amplitude_change, -50, 50, 'amplitude')
        
        # 生成地址向量
        channel_vector = self.generate_address_vector(channel, 'channels')
        code_vector = self.generate_address_vector(code, 'codes')
        
        # 特征融合：将所有特征向量连接成一个长向量
        feature_vector = torch.cat([
            channel_vector,
            code_vector,
            depth_vector,
            amplitude_vector,
            depth_change_vector,
            amplitude_change_vector
        ])
        
        return feature_vector
    
    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        return torchhd.cosine_similarity(vec1, vec2)


class TorchHDClassifier:
    """基于NeuralHD的超维计算分类器"""
    
    def __init__(self, encoder, n_classes=15, device="cpu"):
        """
        初始化NeuralHD分类器
        
        Args:
            encoder: TorchHDEncoder实例
            n_classes: 类别数量（根据需求文档，共15类）
            device: 设备（cpu或cuda）
        """
        self.encoder = encoder
        self.device = device
        self.n_classes = n_classes
        
        # 初始化NeuralHD模型
        # NeuralHD参数：
        # n_features: 输入特征数量（encoder.n_features * encoder.d）
        # n_dimensions: 超维向量维度
        # n_classes: 类别数量
        self.classifier = NeuralHD(
            n_features=encoder.n_features * encoder.d, 
            n_dimensions=encoder.d, 
            n_classes=n_classes, 
            device=device
        )
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # 类别映射
        self.class_map = {
            # 伤损类型
            "轨头外侧核伤": 0,
            "轨头中心核伤": 1,
            "轨头内侧核伤": 2,
            "轨头内、外、中心核伤": 3,
            "轨头内、中心核伤": 4,
            "轨头外、中心核伤": 5,
            "轨头水平分裂": 6,
            "轨头、轨腰分裂": 7,
            "轨腰水平裂": 8,
            "轨底断裂": 9,
            "螺孔裂纹": 10,
            "其他伤损": 11,
            # 正常结构类型
            "接头": 12,
            "螺孔": 13,
            "导线孔": 14,
            "焊缝": 15,
            "鱼鳞伤": 16,
            "轨底锈蚀": 17,
            "轨底焊筋": 18,
            "道岔": 19,
            "其他": 20
        }
        
        # 反向映射
        self.reverse_class_map = {v: k for k, v in self.class_map.items()}
    
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
    
    def train(self, samples, labels, epochs=50, batch_size=32):
        """
        训练NeuralHD分类器
        
        Args:
            samples: 样本列表，每个样本为(channel, code, depth, amplitude)
            labels: 对应标签列表
            epochs: 训练轮数
            batch_size: 批次大小
        """
        # 将样本转换为特征向量和张量
        features = []
        target_labels = []
        
        for sample, label in zip(samples, labels):
            channel, code, depth, amplitude = sample
            
            # 编码样本
            feature_vector = self.encoder.encode_sample(channel, code, depth, amplitude)
            features.append(feature_vector)
            
            # 转换标签为索引
            target_labels.append(self.class_map.get(label, 20))  # 默认其他类
        
        # 转换为张量
        features_tensor = torch.stack(features).to(self.device)
        labels_tensor = torch.tensor(target_labels, dtype=torch.long).to(self.device)
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in dataloader:
                # 前向传播
                outputs = self.classifier(batch_features)
                loss = self.loss_fn(outputs, batch_labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 统计损失和准确率
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            # 打印每轮的损失和准确率
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    def predict(self, sample):
        """
        预测样本类别
        
        Args:
            sample: 待预测样本 (channel, code, depth, amplitude)
            
        Returns:
            预测标签，相似度字典
        """
        # 编码样本
        channel, code, depth, amplitude = sample
        feature_vector = self.encoder.encode_sample(channel, code, depth, amplitude)
        
        # 转换为批量张量
        feature_tensor = feature_vector.unsqueeze(0).to(self.device)
        
        # 预测
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(feature_tensor)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        
        # 转换为类别标签
        predicted_label = self.reverse_class_map[predicted.item()]
        
        # 构建相似度字典
        similarities = {}
        for i, prob in enumerate(probabilities):
            class_name = self.reverse_class_map[i]
            similarities[class_name] = prob.item()
        
        return predicted_label, similarities
    
    def retrain(self, samples, labels, epochs=10, batch_size=32, learning_rate=0.0001):
        """
        重训练优化
        
        Args:
            samples: 训练样本
            labels: 训练标签
            epochs: 重训练迭代次数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        # 更新优化器学习率
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        
        # 调用训练方法进行重训练
        self.train(samples, labels, epochs=epochs, batch_size=batch_size)


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
if __name__ == "__main__":
    # 1. 初始化编码器
    print("初始化编码器...")
    encoder = TorchHDEncoder(d=10000, n_features=4, seed=42, device="cpu")
    
    # 2. 准备数据
    print("准备训练数据...")
    processor = BScanDataProcessor(encoder)
    samples, labels = processor.create_training_data(num_samples=500)
    
    # 转换为训练格式（不包含db_value，保持原有格式）
    training_samples = [(s[0], s[1], s[2], s[3]) for s in samples]
    
    # 3. 训练分类器
    print("训练分类器...")
    classifier = TorchHDClassifier(encoder, n_classes=21, device="cpu")  # 21个类别（11伤损+9正常+1其他）
    classifier.train(training_samples, labels, epochs=20, batch_size=32)
    
    # 4. 测试
    print("测试模型...")
    test_sample = (1, 0, 125, 80)  # 通道1, 编码值0, 深度125mm, 幅度80%
    predicted_label, similarities = classifier.predict(test_sample)
    
    print(f"测试样本预测结果: {predicted_label}")
    print("各类别相似度:")
    for class_name, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {class_name}: {sim:.4f}")
    
    # 5. 重训练优化
    print("\n开始重训练优化...")
    classifier.retrain(training_samples, labels, epochs=10, batch_size=32, learning_rate=0.0001)
    
    # 6. 重训练后重新测试
    print("\n重训练后测试...")
    predicted_label, similarities = classifier.predict(test_sample)
    print(f"重训练后预测结果: {predicted_label}")
    print("各类别相似度:")
    for class_name, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {class_name}: {sim:.4f}")
