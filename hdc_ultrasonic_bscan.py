import numpy as np
import random
from typing import Dict, List, Tuple

# 超维计算参数
DIMENSION = 10000  # 超维向量维度
SEED = 42  # 随机种子

# 设置随机种子，保证结果可复现
np.random.seed(SEED)
random.seed(SEED)

class BScanDataProcessor:
    """B型显示数据处理器"""
    
    def __init__(self, x_resolution: float = 3.0, y_resolution: float = 1.0):
        """
        初始化B显数据处理器
        
        Args:
            x_resolution: X轴分辨率（mm）
            y_resolution: Y轴分辨率（mm）
        """
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.bscan_matrix = None  # B显数据矩阵
        self.max_channels = 0  # 最大通道数
        self.max_encoding = 0  # 最大编码值
    
    def calculate_y_coordinate(self, depth: float, y_max: float) -> float:
        """
        计算Y坐标
        
        Args:
            depth: 深度值（mm）
            y_max: 最大Y坐标
            
        Returns:
            计算得到的Y坐标
        """
        return y_max * depth / 3
    
    def calculate_x_coordinate(self, current_x: float, offset: float, encoding_value: float) -> float:
        """
        计算X坐标
        
        Args:
            current_x: 当前X坐标
            offset: 拼孔值（偏置参数）
            encoding_value: 编码值（Δx）
            
        Returns:
            计算得到的X坐标
        """
        return current_x + offset + encoding_value
    
    def init_bscan_matrix(self, max_channels: int, max_encoding: int):
        """
        初始化B显数据矩阵
        
        Args:
            max_channels: 最大通道数
            max_encoding: 最大编码值
        """
        self.max_channels = max_channels
        self.max_encoding = max_encoding
        # 初始化B显矩阵，值为None表示无数据
        self.bscan_matrix = [[None for _ in range(max_encoding)] for _ in range(max_channels)]
    
    def update_bscan_matrix(self, channel: int, encoding: int, depth: float):
        """
        更新B显数据矩阵
        
        Args:
            channel: 通道号
            encoding: 编码值
            depth: 深度值
        """
        if 0 <= channel < self.max_channels and 0 <= encoding < self.max_encoding:
            # 新编码值对应产生的深度值自右向左填充
            self.bscan_matrix[channel][encoding] = depth
    
    def get_bscan_data(self) -> List[List[float]]:
        """
        获取B显数据矩阵
        
        Returns:
            B显数据矩阵
        """
        return self.bscan_matrix

class HyperdimensionalComputing:
    """超维计算核心模块"""
    
    def __init__(self, dimension: int = DIMENSION):
        """
        初始化超维计算模块
        
        Args:
            dimension: 超维向量维度
        """
        self.dimension = dimension
        self.item_memory = {}  # 项目存储器：存储地址和数值的超维向量
        self.associative_memory = {}  # 联想存储器：存储类别标签和对应的类别超向量
    
    def generate_random_vector(self) -> np.ndarray:
        """
        生成随机超向量，元素为±1
        
        Returns:
            随机超向量
        """
        return np.random.choice([-1, 1], size=self.dimension)
    
    def encode_address(self, address: str) -> np.ndarray:
        """
        编码地址为超维向量
        
        Args:
            address: 地址字符串
            
        Returns:
            地址超维向量
        """
        if address not in self.item_memory:
            self.item_memory[address] = self.generate_random_vector()
        return self.item_memory[address]
    
    def encode_value(self, value: float, value_range: Tuple[float, float]) -> np.ndarray:
        """
        编码数值为超维向量
        
        Args:
            value: 数值
            value_range: 数值范围（min, max）
            
        Returns:
            数值超维向量
        """
        # 将数值映射到[-1, 1]范围
        normalized_value = 2 * (value - value_range[0]) / (value_range[1] - value_range[0]) - 1
        
        # 生成数值超向量
        value_key = f"value_{value_range[0]}_{value_range[1]}_{value}"
        if value_key not in self.item_memory:
            # 生成基础随机向量
            base_vector = self.generate_random_vector()
            # 根据数值大小调整向量
            scaled_vector = base_vector * normalized_value
            # 标准化处理
            scaled_vector[scaled_vector > 0] = 1
            scaled_vector[scaled_vector <= 0] = -1
            self.item_memory[value_key] = scaled_vector
        
        return self.item_memory[value_key]
    
    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        绑定操作：逐元素乘法
        
        Args:
            vec1: 第一个超向量
            vec2: 第二个超向量
            
        Returns:
            绑定后的超向量
        """
        return np.multiply(vec1, vec2)
    
    def permute(self, vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        置换操作：循环移位
        
        Args:
            vec: 超向量
            shift: 移位量
            
        Returns:
            置换后的超向量
        """
        return np.roll(vec, shift)
    
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        捆绑操作：向量加法
        
        Args:
            vectors: 超向量列表
            
        Returns:
            捆绑后的超向量
        """
        result = np.zeros(self.dimension)
        for vec in vectors:
            result += vec
        return result
    
    def normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        标准化操作：将向量元素转换为±1
        
        Args:
            vec: 超向量
            
        Returns:
            标准化后的超向量
        """
        normalized = np.copy(vec)
        normalized[normalized > 0] = 1
        normalized[normalized <= 0] = -1
        return normalized
    
    def train(self, training_data: Dict[str, List[Dict[str, float]]], value_ranges: Dict[str, Tuple[float, float]]):
        """
        超维计算训练过程
        
        Args:
            training_data: 训练数据，格式为{"类别": [{"特征1": 值1, "特征2": 值2, ...}, ...]}
            value_ranges: 各特征的数值范围，格式为{"特征": (min, max)}
        """
        # 清空联想存储器
        self.associative_memory.clear()
        
        # 对每个类别进行训练
        for class_label, samples in training_data.items():
            class_vectors = []
            
            # 对每个样本进行编码
            for sample in samples:
                sample_vector = np.zeros(self.dimension)
                
                # 对每个特征进行编码
                for feature_name, feature_value in sample.items():
                    if feature_name in value_ranges:
                        # 编码地址（特征名）
                        address_vec = self.encode_address(feature_name)
                        # 编码数值（特征值）
                        value_vec = self.encode_value(feature_value, value_ranges[feature_name])
                        # 绑定地址和数值
                        bound_vec = self.bind(address_vec, value_vec)
                        # 将绑定后的向量添加到样本向量中
                        sample_vector += bound_vec
                
                # 标准化样本向量
                normalized_sample_vec = self.normalize(sample_vector)
                class_vectors.append(normalized_sample_vec)
            
            # 捆绑同一类别的所有样本向量
            class_bundle = self.bundle(class_vectors)
            # 标准化类别向量
            normalized_class_vec = self.normalize(class_bundle)
            # 存储到联想存储器
            self.associative_memory[class_label] = normalized_class_vec
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个超向量的相似度（点积）
        
        Args:
            vec1: 第一个超向量
            vec2: 第二个超向量
            
        Returns:
            相似度值，范围为[-1, 1]
        """
        return np.dot(vec1, vec2) / self.dimension
    
    def test(self, test_sample: Dict[str, float], value_ranges: Dict[str, Tuple[float, float]]) -> Tuple[str, float]:
        """
        超维计算测试过程
        
        Args:
            test_sample: 测试样本，格式为{"特征1": 值1, "特征2": 值2, ...}
            value_ranges: 各特征的数值范围，格式为{"特征": (min, max)}
            
        Returns:
            分类结果（类别标签，最大相似度）
        """
        # 编码测试样本
        test_vector = np.zeros(self.dimension)
        
        for feature_name, feature_value in test_sample.items():
            if feature_name in value_ranges:
                # 编码地址（特征名）
                address_vec = self.encode_address(feature_name)
                # 编码数值（特征值）
                value_vec = self.encode_value(feature_value, value_ranges[feature_name])
                # 绑定地址和数值
                bound_vec = self.bind(address_vec, value_vec)
                # 将绑定后的向量添加到测试向量中
                test_vector += bound_vec
        
        # 标准化测试向量
        normalized_test_vec = self.normalize(test_vector)
        
        # 计算与所有类别超向量的相似度
        max_similarity = -float('inf')
        predicted_class = None
        
        for class_label, class_vec in self.associative_memory.items():
            sim = self.similarity(normalized_test_vec, class_vec)
            if sim > max_similarity:
                max_similarity = sim
                predicted_class = class_label
        
        return predicted_class, max_similarity

class RailFlawClassifier:
    """钢轨伤损分类器"""
    
    def __init__(self):
        """初始化钢轨伤损分类器"""
        self.hdc = HyperdimensionalComputing()
        self.flaw_types = {
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
        
        self.normal_structures = {
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
        
        # 特征范围示例（实际应用中需要根据真实数据调整）
        self.value_ranges = {
            'depth': (0.0, 200.0),  # 深度值范围（mm）
            'encoding': (0, 1000),   # 编码值范围
            'angle': (0, 180),       # 角度值范围（度）
            'offset': (-50, 50),     # 拼孔值范围（mm）
            'channel': (0, 10)       # 通道号范围
        }
    
    def extract_features(self, bscan_data: List[List[float]], channel: int, encoding: int) -> Dict[str, float]:
        """
        从B显数据中提取特征
        
        Args:
            bscan_data: B显数据矩阵
            channel: 通道号
            encoding: 编码值
            
        Returns:
            提取的特征字典
        """
        # 示例特征提取，实际应用中需要根据具体需求设计
        depth = bscan_data[channel][encoding] if bscan_data and channel < len(bscan_data) and encoding < len(bscan_data[channel]) else 0.0
        
        features = {
            'depth': depth,
            'channel': float(channel),
            'encoding': float(encoding),
            # 可以添加更多特征提取
        }
        
        return features
    
    def train_classifier(self, training_data: Dict[str, List[Dict[str, float]]]):
        """
        训练伤损分类器
        
        Args:
            training_data: 训练数据，格式为{"类别": [{"特征1": 值1, "特征2": 值2, ...}, ...]}
        """
        self.hdc.train(training_data, self.value_ranges)
    
    def classify_flaw(self, test_sample: Dict[str, float]) -> Tuple[str, float]:
        """
        对测试样本进行伤损分类
        
        Args:
            test_sample: 测试样本，格式为{"特征1": 值1, "特征2": 值2, ...}
            
        Returns:
            分类结果（类别标签，相似度）
        """
        return self.hdc.test(test_sample, self.value_ranges)
    
    def get_flaw_type_name(self, flaw_code: int) -> str:
        """
        获取伤损类型名称
        
        Args:
            flaw_code: 伤损类型代码
            
        Returns:
            伤损类型名称
        """
        return self.flaw_types.get(flaw_code, "未知伤损类型")
    
    def get_normal_structure_name(self, structure_code: int) -> str:
        """
        获取正常结构名称
        
        Args:
            structure_code: 正常结构代码
            
        Returns:
            正常结构名称
        """
        return self.normal_structures.get(structure_code, "未知正常结构")

# 示例用法
if __name__ == "__main__":
    # 创建B显数据处理器
    bscan_processor = BScanDataProcessor()
    
    # 初始化B显矩阵（示例：10个通道，100个编码值）
    bscan_processor.init_bscan_matrix(max_channels=10, max_encoding=100)
    
    # 模拟更新B显数据
    for channel in range(5):
        for encoding in range(50):
            # 随机生成深度值（0-100mm）
            depth = random.uniform(0, 100)
            bscan_processor.update_bscan_matrix(channel, encoding, depth)
    
    # 创建钢轨伤损分类器
    flaw_classifier = RailFlawClassifier()
    
    # 模拟训练数据（实际应用中需要使用真实数据）
    training_data = {
        "核伤类": [
            {'depth': 50.0, 'channel': 0.0, 'encoding': 10.0},
            {'depth': 60.0, 'channel': 1.0, 'encoding': 20.0},
            {'depth': 55.0, 'channel': 0.0, 'encoding': 30.0}
        ],
        "正常结构": [
            {'depth': 0.0, 'channel': 2.0, 'encoding': 15.0},
            {'depth': 0.0, 'channel': 3.0, 'encoding': 25.0},
            {'depth': 0.0, 'channel': 2.0, 'encoding': 35.0}
        ]
    }
    
    # 训练分类器
    flaw_classifier.train_classifier(training_data)
    
    # 测试分类器
    test_sample = {'depth': 52.0, 'channel': 0.0, 'encoding': 15.0}
    result, similarity = flaw_classifier.classify_flaw(test_sample)
    print(f"分类结果：{result}，相似度：{similarity:.4f}")
    
    # 提取B显数据特征并分类
    print("\nB显数据分类测试：")
    bscan_data = bscan_processor.get_bscan_data()
    for channel in range(3):
        for encoding in range(5, 10):
            features = flaw_classifier.extract_features(bscan_data, channel, encoding)
            result, similarity = flaw_classifier.classify_flaw(features)
            print(f"通道{channel}, 编码{encoding}: 分类结果={result}, 相似度={similarity:.4f}")