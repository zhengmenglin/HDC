import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN

# 配置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 超维计算参数
DIMENSION = 10000  # 超维向量维度
SEED = 42  # 随机种子


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


class BScanPoint:
    """B显数据点类，存储深度值、原始dB值、幅度值和二值化结果"""
    def __init__(self, depth: float = None, db_value: float = None, threshold: float = 30.0, amplitude: float = 100.0):
        """
        初始化B显数据点
        
        Args:
            depth: 深度值（mm）
            db_value: 原始超声信号强度（dB）
            threshold: 二值化阈值（dB）
            amplitude: 信号幅度值（%），默认为100%
        """
        self.depth = depth
        self.db_value = db_value
        self.amplitude = amplitude  # 新添加的幅度值字段
        self.threshold = threshold
        # 计算二值化结果：dB值大于阈值为1，否则为0
        self.binary_value = 1 if (db_value is not None and db_value > threshold) else 0

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
        self.max_channels = 9  # 固定为9个通道
        self.max_encoding = 0  # 最大编码值
        self.threshold = 30.0  # 二值化阈值（dB）
        
        # 初始化9个通道的配置（根据需求文档）
        # 通道配置：(俯仰角度, 水平角度, 拼孔值)
        self.channel_configs = [
            ChannelConfig(70, 15, 10),   # 通道1：俯仰70°, 水平角度α(默认15°), 拼孔值10
            ChannelConfig(70, 15, 60),   # 通道2：俯仰70°, 水平角度α(默认15°), 拼孔值60
            ChannelConfig(70, 15, 20),   # 通道3：俯仰70°, 水平角度α(默认15°), 拼孔值20
            ChannelConfig(70, 15, 50),   # 通道4：俯仰70°, 水平角度α(默认15°), 拼孔值50
            ChannelConfig(70, 0, 30),    # 通道5：俯仰70°, 水平角度0°, 拼孔值30
            ChannelConfig(70, 0, 40),    # 通道6：俯仰70°, 水平角度0°, 拼孔值40
            ChannelConfig(37, 0, 0),     # 通道7：俯仰37°, 水平角度0°, 拼孔值0
            ChannelConfig(37, 0, 70),     # 通道8：俯仰37°, 水平角度0°, 拼孔值70
            ChannelConfig(0, 0, 80)       # 通道9：俯仰0°, 水平角度0°, 拼孔值80
        ]
    
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
    
    def calculate_x_coordinate(self, current_x: float, channel: int, encoding_value: float) -> float:
        """
        计算X坐标
        
        Args:
            current_x: 当前X坐标
            channel: 通道号
            encoding_value: 编码值（Δx）
            
        Returns:
            计算得到的X坐标
        """
        if 0 <= channel < self.max_channels:
            offset = self.channel_configs[channel].offset
            return current_x + offset + encoding_value
        return current_x
    
    def init_bscan_matrix(self, max_encoding: int):
        """
        初始化B显数据矩阵
        
        Args:
            max_encoding: 最大编码值
        """
        self.max_encoding = max_encoding
        # 初始化B显矩阵，每个元素为BScanPoint对象
        self.bscan_matrix = [[BScanPoint(threshold=self.threshold) for _ in range(max_encoding)] for _ in range(self.max_channels)]
    
    def update_bscan_matrix(self, channel: int, encoding: int, depth: float, db_value: float):
        """
        更新B显数据矩阵
        
        Args:
            channel: 通道号
            encoding: 编码值
            depth: 深度值（mm）
            db_value: 原始超声信号强度（dB）
        """
        if 0 <= channel < self.max_channels and 0 <= encoding < self.max_encoding:
            # 更新BScanPoint对象
            self.bscan_matrix[channel][encoding] = BScanPoint(depth, db_value, self.threshold)
    
    def get_bscan_data(self, data_type: str = "depth") -> List[List[float]]:
        """
        获取B显数据矩阵
        
        Args:
            data_type: 数据类型，可选值："depth"（深度值）、"db"（原始dB值）、"binary"（二值化结果）
            
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
                elif data_type == "binary":
                    value = point.binary_value
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
    
    def visualize_bscan(self, title: str = "B型显示图", data_type: str = "depth", 
                        clusters: List[Dict] = None, cluster_classifications: Dict = None):
        """
        可视化B显数据矩阵，带伤损画框和相似度显示
        
        Args:
            title: 图像标题
            data_type: 数据类型，可选值："depth"（深度值）、"db"（原始dB值）、"binary"（二值化结果）
            clusters: 伤损聚类结果列表
            cluster_classifications: 聚类分类结果，格式为 {cluster_id: (class_label, similarity, all_similarities)}
        """
        if self.bscan_matrix is None:
            print("B显数据矩阵尚未初始化")
            return
        
        # 提取对应的数据类型
        bscan_data = self.get_bscan_data(data_type)
        bscan_array = np.array(bscan_data, dtype=float)
        
        # 创建图像和子图布局
        # 主图 + 右侧相似度子图
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])
        
        # 主图
        ax_main = fig.add_subplot(gs[0, 0])
        
        # 颜色列表，9个不同颜色
        colors = plt.cm.rainbow(np.linspace(0, 1, self.max_channels))
        
        if data_type in ["depth", "db"]:
            # 深度值和dB值：使用散点图
            # 横轴为编码值，纵轴为深度值或dB值
            
            for channel in range(self.max_channels):
                # 提取该通道的数据
                values = bscan_array[channel, :]
                # 编码值范围
                encoding_values = np.arange(self.max_encoding)
                
                # 绘制散点图
                ax_main.scatter(encoding_values, values, color=colors[channel], 
                        label=f'通道{channel+1}', s=20, alpha=0.7)
            
            # 设置标题和标签
            ax_main.set_title(title)
            ax_main.set_xlabel('编码值')
            
            if data_type == "depth":
                ax_main.set_ylabel('深度值 (mm)')
                # 设置深度值原点在右上角，深度零点在上面
                ax_main.invert_yaxis()
            else:
                ax_main.set_ylabel('信号强度 (dB)')
            
            # 设置坐标轴刻度
            ax_main.set_xticks(np.arange(0, self.max_encoding, step=10))
            
            # 添加图例
            ax_main.legend(loc='best')
            
            # 添加网格线
            ax_main.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制伤损框
            if clusters and cluster_classifications:
                # 为每个聚类绘制边界框和标注
                for cluster in clusters:
                    channel = cluster['channel']
                    bounding_box = cluster['bounding_box']
                    cluster_id = cluster['cluster_id']
                    
                    # 获取分类结果
                    if cluster_id in cluster_classifications:
                        class_label, similarity, _ = cluster_classifications[cluster_id]
                        
                        # 绘制边界框
                        min_encoding, max_encoding, min_depth, max_depth = bounding_box
                        rect = plt.Rectangle((min_encoding, min_depth), 
                                           max_encoding - min_encoding, 
                                           max_depth - min_depth, 
                                           linewidth=2, 
                                           edgecolor='red', 
                                           facecolor='none', 
                                           alpha=0.8)
                        ax_main.add_patch(rect)
                        
                        # 标注伤损类型和相似度
                        ax_main.text(min_encoding, min_depth - 2, 
                                    f'{class_label} ({similarity:.2f})', 
                                    fontsize=9, 
                                    fontweight='bold', 
                                    color='red',
                                    bbox=dict(facecolor='white', alpha=0.8))
        else:  # binary
            # 二值化结果：保持原有热图可视化方式
            im = ax_main.imshow(bscan_array, cmap='binary', aspect='auto', origin='upper')
            
            # 设置颜色条标签
            colorbar_label = '二值化结果'
            
            # 添加颜色条
            fig.colorbar(im, ax=ax_main, label=colorbar_label)
            
            # 设置标题和标签
            ax_main.set_title(title)
            ax_main.set_xlabel('编码值')
            ax_main.set_ylabel('通道号')
            
            # 设置坐标轴刻度
            ax_main.set_xticks(np.arange(0, self.max_encoding, step=10))
            ax_main.set_yticks(np.arange(0, self.max_channels, step=1))
            
            # 添加网格线
            ax_main.grid(True, linestyle='--', alpha=0.7)
        
        # 右侧相似度子图
        if cluster_classifications and data_type in ["depth", "db"]:
            ax_similarity = fig.add_subplot(gs[0, 1])
            
            # 收集所有相似度不为零的类别
            all_similarities = {}
            for cluster_id, (_, _, sims) in cluster_classifications.items():
                for class_label, sim in sims.items():
                    if sim != 0:
                        if class_label not in all_similarities:
                            all_similarities[class_label] = 0
                        all_similarities[class_label] += sim
            
            # 按相似度排序
            sorted_classes = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_classes:
                labels = [item[0] for item in sorted_classes]
                values = [item[1] for item in sorted_classes]
                
                # 绘制相似度条形图（垂直方向）
                ax_similarity.barh(range(len(labels)), values, color=plt.cm.rainbow(np.linspace(0, 1, len(labels))))
                ax_similarity.set_yticks(range(len(labels)))
                ax_similarity.set_yticklabels(labels, fontsize=8)
                ax_similarity.set_xlabel('相似度')
                ax_similarity.set_title('各类别相似度')
                ax_similarity.grid(True, linestyle='--', alpha=0.7)
            else:
                ax_similarity.text(0.5, 0.5, '无相似度数据', ha='center', va='center', fontsize=10)
                ax_similarity.set_title('各类别相似度')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{title.replace(' ', '_')}_{data_type}.png", dpi=300, bbox_inches='tight')
        print(f"B显图像已保存到: {title.replace(' ', '_')}_{data_type}.png")
        
        # 显示图像
        plt.show()
    
    def cluster_flaws(self, eps: float = 5.0, min_samples: int = 2) -> List[Dict]:
        """
        使用DBSCAN算法对伤损点进行聚类
        
        Args:
            eps: DBSCAN算法的邻域半径
            min_samples: 邻域内最小样本数
            
        Returns:
            聚类结果列表，每个聚类包含：
            - channel: 通道号
            - bounding_box: 边界框 (min_encoding, max_encoding, min_depth, max_depth)
            - points: 包含的点列表 [(encoding, depth), ...]
            - cluster_id: 聚类ID
        """
        clusters = []
        
        # 遍历每个通道
        for channel in range(self.max_channels):
            # 收集该通道的所有伤损点 (binary_value == 1)
            flaw_points = []
            for encoding in range(self.max_encoding):
                point = self.bscan_matrix[channel][encoding]
                if point.binary_value == 1:
                    flaw_points.append((encoding, point.depth))
            
            if not flaw_points:
                continue
            
            # 将点转换为numpy数组
            X = np.array(flaw_points)
            
            # 使用DBSCAN进行聚类
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 处理每个聚类
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # 噪声点，跳过
                    continue
                
                # 获取该聚类的所有点
                cluster_points = X[labels == label]
                if len(cluster_points) < min_samples:
                    continue
                
                # 计算边界框
                min_encoding = np.min(cluster_points[:, 0])
                max_encoding = np.max(cluster_points[:, 0])
                min_depth = np.min(cluster_points[:, 1])
                max_depth = np.max(cluster_points[:, 1])
                
                # 添加聚类结果
                clusters.append({
                    'channel': channel,
                    'bounding_box': (min_encoding, max_encoding, min_depth, max_depth),
                    'points': cluster_points.tolist(),
                    'cluster_id': label
                })
        
        return clusters
    
    def visualize_channel_config(self):
        """
        可视化通道配置参数
        """
        # 创建图像
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # 通道号列表
        channels = np.arange(1, self.max_channels + 1)
        
        # 1. 俯仰角度分布
        pitch_angles = [config.pitch_angle for config in self.channel_configs]
        axes[0].bar(channels, pitch_angles, color='blue')
        axes[0].set_title('各通道俯仰角度分布')
        axes[0].set_xlabel('通道号')
        axes[0].set_ylabel('俯仰角度 (度)')
        axes[0].set_xticks(channels)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. 水平角度分布
        horizontal_angles = [config.horizontal_angle for config in self.channel_configs]
        axes[1].bar(channels, horizontal_angles, color='green')
        axes[1].set_title('各通道水平角度分布')
        axes[1].set_xlabel('通道号')
        axes[1].set_ylabel('水平角度 (度)')
        axes[1].set_xticks(channels)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. 拼孔值分布
        offsets = [config.offset for config in self.channel_configs]
        axes[2].bar(channels, offsets, color='red')
        axes[2].set_title('各通道拼孔值分布')
        axes[2].set_xlabel('通道号')
        axes[2].set_ylabel('拼孔值 (mm)')
        axes[2].set_xticks(channels)
        axes[2].grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('channel_config_visualization.png', dpi=300, bbox_inches='tight')
        print('通道配置可视化图像已保存到: channel_config_visualization.png')
        
        # 显示图像
        plt.show()

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
        
        # 11种伤损类型（根据需求更新）
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
            12: "其他伤损"  # 无法识别的伤损类型
        }
        
        # 8种正常结构类型（根据需求更新）
        self.normal_structures = {
            1: "接头",
            2: "螺孔",
            3: "导线孔",
            4: "焊缝",
            5: "鱼鳞伤",
            6: "轨底锈蚀",
            7: "轨底焊筋",
            8: "道岔",
            9: "其他"  # 无法识别的正常结构
        }
        
        # 特征范围（根据实际数据调整）
        self.value_ranges = {
            'depth': (0.0, 200.0),  # 深度值范围（mm）
            'encoding': (0, 1000),   # 编码值范围
            'angle': (0, 180),       # 角度值范围（度）
            'offset': (-50, 100),     # 拼孔值范围（mm）
            'channel': (0, 9),       # 通道号范围（0-8，对应9个通道）
            'db_value': (-20, 100),  # 原始dB值范围
            'binary_value': (0, 1)    # 二值化结果
        }
    
    def extract_features(self, bscan_data: List[List['BScanPoint']], channel: int, encoding: int) -> Dict[str, float]:
        """
        从B显数据中提取特征
        
        Args:
            bscan_data: B显数据矩阵（BScanPoint对象列表）
            channel: 通道号
            encoding: 编码值
            
        Returns:
            提取的特征字典
        """
        if not bscan_data or channel >= len(bscan_data) or encoding >= len(bscan_data[channel]):
            # 无效数据，返回默认值
            return {
                'depth': 0.0,
                'db_value': 0.0,
                'binary_value': 0.0,
                'channel': float(channel),
                'encoding': float(encoding)
            }
        
        # 获取BScanPoint对象
        point = bscan_data[channel][encoding]
        
        # 提取特征
        features = {
            'depth': point.depth if point.depth is not None else 0.0,
            'db_value': point.db_value if point.db_value is not None else 0.0,
            'binary_value': float(point.binary_value),
            'channel': float(channel),
            'encoding': float(encoding)
        }
        
        return features
    
    def train_classifier(self, training_data: Dict[str, List[Dict[str, float]]]):
        """
        训练伤损分类器
        
        Args:
            training_data: 训练数据，格式为{"类别": [{"特征1": 值1, "特征2": 值2, ...}, ...]}
        """
        self.hdc.train(training_data, self.value_ranges)
    
    def classify_flaw(self, test_sample: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        对测试样本进行伤损分类
        
        Args:
            test_sample: 测试样本，格式为{"特征1": 值1, "特征2": 值2, ...}
            
        Returns:
            分类结果：(类别标签, 最大相似度, 所有类别相似度字典)
        """
        # 获取测试样本与所有类别超向量的相似度
        similarities = {}
        for class_label, class_vec in self.hdc.associative_memory.items():
            # 编码测试样本
            test_vector = np.zeros(self.hdc.dimension)
            for feature_name, feature_value in test_sample.items():
                if feature_name in self.value_ranges:
                    address_vec = self.hdc.encode_address(feature_name)
                    value_vec = self.hdc.encode_value(feature_value, self.value_ranges[feature_name])
                    bound_vec = self.hdc.bind(address_vec, value_vec)
                    test_vector += bound_vec
            test_vector = self.hdc.normalize(test_vector)
            
            # 计算相似度
            sim = self.hdc.similarity(test_vector, class_vec)
            similarities[class_label] = sim
        
        # 找到最大相似度对应的类别
        max_similarity = -float('inf')
        predicted_class = "其他"
        
        for class_label, sim in similarities.items():
            if sim > max_similarity:
                max_similarity = sim
                predicted_class = class_label
        
        return predicted_class, max_similarity, similarities
    
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
    
    def visualize_classification(self, test_sample: Dict[str, float]):
        """
        可视化分类结果
        
        Args:
            test_sample: 测试样本
        """
        # 计算测试样本与所有类别超向量的相似度
        similarities = {}
        for class_label, class_vec in self.hdc.associative_memory.items():
            # 编码测试样本
            test_vector = np.zeros(self.hdc.dimension)
            for feature_name, feature_value in test_sample.items():
                if feature_name in self.value_ranges:
                    address_vec = self.hdc.encode_address(feature_name)
                    value_vec = self.hdc.encode_value(feature_value, self.value_ranges[feature_name])
                    bound_vec = self.hdc.bind(address_vec, value_vec)
                    test_vector += bound_vec
            test_vector = self.hdc.normalize(test_vector)
            
            # 计算相似度
            sim = self.hdc.similarity(test_vector, class_vec)
            similarities[class_label] = sim
        
        # 创建图像
        plt.figure(figsize=(12, 8))
        
        # 绘制相似度条形图
        class_labels = list(similarities.keys())
        similarity_values = list(similarities.values())
        
        # 按相似度值排序
        sorted_indices = np.argsort(similarity_values)[::-1]  # 降序排列
        sorted_class_labels = [class_labels[i] for i in sorted_indices]
        sorted_similarity_values = [similarity_values[i] for i in sorted_indices]
        
        # 生成颜色列表
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_labels)))
        
        plt.bar(sorted_class_labels, sorted_similarity_values, color=colors)
        plt.title('测试样本与各类别的相似度')
        plt.xlabel('类别')
        plt.ylabel('相似度')
        plt.ylim(-1, 1)
        
        # 在条形图上显示数值
        for i, v in enumerate(sorted_similarity_values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', rotation=45)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 旋转x轴标签，避免重叠
        plt.xticks(rotation=45, ha='right')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('classification_similarity.png', dpi=300, bbox_inches='tight')
        print("分类相似度图像已保存到: classification_similarity.png")
        
        # 显示图像
        plt.show()
    
    def visualize_feature_distribution(self, training_data: Dict[str, List[Dict[str, float]]]):
        """
        可视化训练数据的特征分布
        
        Args:
            training_data: 训练数据
        """
        # 提取所有特征
        all_features = set()
        for samples in training_data.values():
            for sample in samples:
                all_features.update(sample.keys())
        
        # 过滤掉不需要可视化的特征
        features_to_visualize = [feature for feature in all_features if feature not in ['binary_value']]
        
        # 为每个特征创建子图
        fig, axes = plt.subplots(len(features_to_visualize), 1, figsize=(14, 5 * len(features_to_visualize)))
        if len(features_to_visualize) == 1:
            axes = [axes]
        
        # 绘制每个特征的分布
        for i, feature in enumerate(features_to_visualize):
            ax = axes[i]
            
            # 收集所有类别的特征值
            feature_values = {}
            for class_label, samples in training_data.items():
                values = [sample[feature] for sample in samples if feature in sample]
                feature_values[class_label] = values
            
            # 绘制箱线图
            box_plot = ax.boxplot(feature_values.values(), labels=feature_values.keys(), patch_artist=True)
            
            # 设置颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(feature_values)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            # 设置标题和标签
            ax.set_title(f'{feature}特征分布', fontsize=14, fontweight='bold')
            ax.set_xlabel('类别', fontsize=12)
            ax.set_ylabel(feature, fontsize=12)
            ax.tick_params(axis='x', rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
        print("特征分布图像已保存到: feature_distribution.png")
        
        # 显示图像
        plt.show()
        
    def visualize_test_samples(self, test_data: List[Dict[str, float]], test_results: List[Tuple[str, float, Dict[str, float]]]):
        """
        可视化测试样本的分类结果
        
        Args:
            test_data: 测试样本列表
            test_results: 测试结果列表，每个元素为 (class_label, max_similarity, all_similarities)
        """
        try:
            # 创建图像
            fig = plt.figure(figsize=(15, 10))
            
            # 提取测试样本的特征
            depths = [sample.get('depth', 0.0) for sample in test_data]
            encodings = [sample.get('encoding', 0.0) for sample in test_data]
            db_values = [sample.get('db_value', 0.0) for sample in test_data]
            
            # 提取分类结果
            class_labels = [result[0] for result in test_results]
            similarities = [result[1] for result in test_results]
            
            # 创建子图布局
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            
            # 1. 深度值 vs 编码值，颜色表示相似度
            ax1 = fig.add_subplot(gs[0, 0])
            scatter1 = ax1.scatter(encodings, depths, c=similarities, cmap='rainbow', s=50, alpha=0.7)
            ax1.set_title('测试样本深度值分布（颜色表示相似度）')
            ax1.set_xlabel('编码值')
            ax1.set_ylabel('深度值 (mm)')
            ax1.invert_yaxis()  # 深度值原点在右上角
            ax1.grid(True, linestyle='--', alpha=0.7)
            fig.colorbar(scatter1, ax=ax1, label='相似度')
            
            # 2. dB值 vs 编码值，颜色表示相似度
            ax2 = fig.add_subplot(gs[0, 1])
            scatter2 = ax2.scatter(encodings, db_values, c=similarities, cmap='rainbow', s=50, alpha=0.7)
            ax2.set_title('测试样本dB值分布（颜色表示相似度）')
            ax2.set_xlabel('编码值')
            ax2.set_ylabel('信号强度 (dB)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            fig.colorbar(scatter2, ax=ax2, label='相似度')
            
            # 3. 分类结果分布
            ax3 = fig.add_subplot(gs[1, 0])
            # 处理字符串标签，转换为numpy数组
            unique_labels = list(set(class_labels))
            label_counts = [class_labels.count(label) for label in unique_labels]
            ax3.bar(unique_labels, label_counts, color=plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))
            ax3.set_title('测试样本分类结果分布')
            ax3.set_xlabel('类别')
            ax3.set_ylabel('样本数')
            ax3.tick_params(axis='x', rotation=45, ha='right')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # 4. 相似度分布
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax4.set_title('测试样本相似度分布')
            ax4.set_xlabel('相似度')
            ax4.set_ylabel('样本数')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            plt.savefig('test_samples_visualization.png', dpi=300, bbox_inches='tight')
            print("测试样本可视化图像已保存到: test_samples_visualization.png")
            
            # 关闭图像，不显示
            plt.close(fig)
        except Exception as e:
            print(f"可视化测试样本时出错: {e}")

# 导入数据读取模块
from data_reader import HexDataReader, DataPoint

# 示例用法
if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="超维计算超声波探伤算法")
    parser.add_argument('--train_data', type=str, default=None, help='训练数据文件路径（16进制）')
    parser.add_argument('--train_labels', type=str, default=None, help='训练数据标签文件路径')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据文件路径（16进制）')
    parser.add_argument('--use_simulated', action='store_true', help='使用模拟数据')
    args = parser.parse_args()
    
    # 创建B显数据处理器
    bscan_processor = BScanDataProcessor()
    
    # 初始化B显矩阵（示例：9个通道，100个编码值）
    bscan_processor.init_bscan_matrix(max_encoding=100)
    
    # 数据点列表
    data_points = []
    
    # 必须提供训练数据文件和标签文件
    if args.train_data and args.train_labels:
        # 从文件读取数据
        print("\n从文件读取训练数据...")
        reader = HexDataReader()
        data_points = reader.read_data_file(args.train_data)
        print(f"读取到 {len(data_points)} 个数据点")
        
        # 更新B显矩阵
        for data_point in data_points:
            if 0 <= data_point.channel < bscan_processor.max_channels and 0 <= data_point.encoding < bscan_processor.max_encoding:
                bscan_processor.update_bscan_matrix(
                    data_point.channel, 
                    data_point.encoding, 
                    data_point.depth, 
                    data_point.db_value
                )
    else:
        print("\n必须提供训练数据文件和标签文件")
        exit(1)
    
    # 可视化通道配置
    print("\n可视化通道配置...")
    bscan_processor.visualize_channel_config()
    
    # 创建钢轨伤损分类器
    flaw_classifier = RailFlawClassifier()
    
    # 训练数据读取
    training_data = {}
    
    if args.train_data and args.train_labels:
        # 从文件读取训练数据
        print("\n从文件读取训练数据...")
        reader = HexDataReader()
        training_data = reader.convert_to_training_data(data_points, args.train_labels)
        print(f"转换为 {len(training_data)} 类训练数据")
    else:
        print("\n未提供训练数据和标签文件，无法进行训练")
        exit(1)
    
    # 训练分类器
    print("\n训练分类器...")
    flaw_classifier.train_classifier(training_data)
    
    # 从文件读取测试数据
    test_data = []
    classification_results = {}
    
    if args.test_data:
        print(f"\n从文件读取测试数据: {args.test_data}")
        reader = HexDataReader()
        test_data_points = reader.read_data_file(args.test_data)
        test_data = reader.convert_to_test_data(test_data_points)
        print(f"读取到 {len(test_data)} 个测试样本")
        
        # 测试所有测试样本
        print("\n测试数据分类结果：")
        test_results = []  # 保存测试结果
        for i, sample in enumerate(test_data):
            result, max_sim, all_sims = flaw_classifier.classify_flaw(sample)
            test_results.append((result, max_sim, all_sims))  # 保存结果
            print(f"测试样本{i+1}: 分类结果={result}, 最大相似度={max_sim:.4f}")
            
            # 记录分类结果
            if result not in classification_results:
                classification_results[result] = 0
            classification_results[result] += 1
        
        # 可视化测试样本
        print("\n可视化测试样本...")
        flaw_classifier.visualize_test_samples(test_data, test_results)
    else:
        print("\n未提供测试数据文件，无法进行测试")
        exit(1)
    
    # 伤损聚类
    print("\n进行伤损聚类...")
    clusters = bscan_processor.cluster_flaws(eps=5.0, min_samples=2)
    print(f"找到 {len(clusters)} 个伤损聚类")
    
    # 对每个聚类进行分类
    cluster_classifications = {}
    for cluster in clusters:
        # 提取聚类中所有点的特征
        cluster_points = cluster['points']
        channel = cluster['channel']
        
        # 对聚类中每个点进行分类，取最大相似度作为聚类的分类结果
        cluster_max_similarity = -float('inf')
        cluster_best_class = "其他"
        cluster_all_similarities = {}
        
        for point in cluster_points:
            encoding, depth = point
            # 提取特征
            features = flaw_classifier.extract_features(
                bscan_processor.get_bscan_points(), channel, int(encoding)
            )
            # 分类
            result, max_sim, all_sims = flaw_classifier.classify_flaw(features)
            
            # 更新聚类的最大相似度和最佳类别
            if max_sim > cluster_max_similarity:
                cluster_max_similarity = max_sim
                cluster_best_class = result
            
            # 合并所有相似度
            for class_label, sim in all_sims.items():
                if class_label not in cluster_all_similarities:
                    cluster_all_similarities[class_label] = 0
                cluster_all_similarities[class_label] += sim
        
        # 保存聚类分类结果
        cluster_classifications[cluster['cluster_id']] = (
            cluster_best_class, cluster_max_similarity, cluster_all_similarities
        )
    
    # 可视化B显数据（深度值），带伤损画框和相似度显示
    print("\n可视化B显数据（深度值）...")
    bscan_processor.visualize_bscan(
        title="超声波探伤B型显示图", 
        data_type="depth",
        clusters=clusters,
        cluster_classifications=cluster_classifications
    )
    
    # 可视化B显数据（dB值），带伤损画框和相似度显示
    print("\n可视化B显数据（dB值）...")
    bscan_processor.visualize_bscan(
        title="超声波探伤B型显示图", 
        data_type="db",
        clusters=clusters,
        cluster_classifications=cluster_classifications
    )
    
    # 可视化B显数据（二值化结果）
    print("\n可视化B显数据（二值化结果）...")
    bscan_processor.visualize_bscan(title="超声波探伤B型显示图", data_type="binary")
    
    # 可视化分类结果统计
    print("\n分类结果统计：")
    for result, count in classification_results.items():
        print(f"{result}: {count}个样本")
    
    # 创建分类结果统计图像
    plt.figure(figsize=(10, 8))
    result_labels = list(classification_results.keys())
    result_counts = list(classification_results.values())
    
    # 绘制饼图
    plt.pie(result_counts, labels=result_labels, autopct='%1.1f%%', startangle=90, 
            wedgeprops={'edgecolor': 'w', 'linewidth': 1})
    plt.title('B显数据分类结果统计')
    plt.axis('equal')  # 确保饼图为圆形
    
    # 保存图像
    plt.savefig('classification_results_stat.png', dpi=300, bbox_inches='tight')
    print("分类结果统计图像已保存到: classification_results_stat.png")
    
    # 绘制分类结果条形图
    plt.figure(figsize=(12, 6))
    plt.bar(result_labels, result_counts, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta'])
    plt.title('B显数据分类结果条形图')
    plt.xlabel('类别')
    plt.ylabel('样本数')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 在条形图上显示数值
    for i, v in enumerate(result_counts):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('classification_results_bar.png', dpi=300, bbox_inches='tight')
    print("分类结果条形图已保存到: classification_results_bar.png")
    
    plt.show()