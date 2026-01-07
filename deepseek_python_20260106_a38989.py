import numpy as np
import json
from collections import defaultdict
import random

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
        # 生成最小值向量
        base_vector = self.generate_binary_hypervector()
        self.item_memory[base_key][min_val] = base_vector
        
        # 计算每次翻转的位数
        flip_bits = self.d // (2 * (num_levels - 1))
        
        # 生成连续映射的数值向量
        for i in range(1, num_levels):
            # 从前一个向量翻转固定位数
            prev_vector = self.item_memory[base_key][min_val + (i-1)]
            flip_indices = np.random.choice(self.d, flip_bits, replace=False)# 随机选择翻转位
            new_vector = prev_vector.copy()
            new_vector[flip_indices] *= -1  # 翻转位
            
            value = min_val + i
            self.item_memory[base_key][value] = new_vector
            
        return self.item_memory[base_key]
    
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
    
    def hamming_similarity(self, vec1, vec2):
        """计算汉明相似度（归一化）"""
        # 对于二元向量(-1,1)，汉明相似度 = (1 - 汉明距离/d)
        matches = np.sum(vec1 == vec2)
        return matches / self.d
    
    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度（适用于实数向量）"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)# 向量1的范数
        norm2 = np.linalg.norm(vec2)# 向量2的范数
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
    
    def retrain(self, samples, labels, iterations=10):
        """
        重训练优化（简化版蒙版重训练）
        Args:
            samples: 训练样本
            labels: 训练标签
            iterations: 重训练迭代次数
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
            
            # 调整类别超向量
            for idx, sample, true_label, wrong_label in misclassified:
                # 编码错误样本
                channel, code, depth, amplitude = sample
                sample_vector = self.encoder.encode_sample(channel, code, depth, amplitude)
                
                # 从错误类别中移除，添加到正确类别
                # 注意：这里简化处理，实际重训练应使用非二元向量和自适应学习率
                if wrong_label in self.class_vectors:
                    self.class_vectors[wrong_label] = self.encoder.bundle(
                        [self.class_vectors[wrong_label], -sample_vector], 
                        normalize=True
                    )
                
                if true_label in self.class_vectors:
                    self.class_vectors[true_label] = self.encoder.bundle(
                        [self.class_vectors[true_label], sample_vector], 
                        normalize=True
                    )
            
            print(f"迭代 {iteration+1}: 错误样本数 {len(misclassified)}")


class BScanDataProcessor:
    """B显数据处理器"""
    
    def __init__(self, encoder):
        self.encoder = encoder
        
    def load_json_data(self, json_file):
        """从JSON文件加载B显数据"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        samples = []
        labels = []  # 假设数据已标注
        
        for key, value in data.items():
            # 解析JSON格式: {"B1":[1,125,80]}
            channel, depth, amplitude = value
            code = 0  # 简化处理，实际应根据编码值
            
            samples.append((channel, code, depth, amplitude))
            
            # 这里简化标签生成，实际应从数据中获取
            label = self._determine_label(channel, depth, amplitude)
            labels.append(label)
        
        return samples, labels
    
    def _determine_label(self, channel, depth, amplitude):
        """根据通道和深度确定标签（简化版）"""
        # 这里应根据实际伤损分类规则实现
        # 示例：根据通道和深度范围判断伤损类型
        if channel in [1, 2, 3]:  # 直70°和斜70°通道
            if depth > 100 and amplitude > 70:
                return "核伤类"
        elif channel in [4, 5]:  # 37°和0°通道
            if depth > 120 and amplitude > 60:
                return "水平裂纹类"
        
        return "正常结构"
    
    def create_training_data(self, num_samples=1000):
        """生成模拟训练数据"""
        samples = []
        labels = []
        
        # 正常结构数据
        for _ in range(num_samples // 2):
            channel = random.randint(1, 7)
            code = random.randint(0, 100)
            depth = random.randint(50, 150)
            amplitude = random.randint(10, 50)
            samples.append((channel, code, depth, amplitude))
            labels.append("正常结构")
        
        # 伤损数据
        for _ in range(num_samples // 2):
            channel = random.randint(1, 7)
            code = random.randint(0, 100)
            depth = random.randint(100, 200)
            amplitude = random.randint(60, 100)
            samples.append((channel, code, depth, amplitude))
            labels.append("伤损类")
        
        return samples, labels


# 使用示例
def main():
    # 1. 初始化编码器
    encoder = HyperdimensionalEncoder(d=10000)  # 使用5000维以降低计算量
    
    # 2. 使用连续映射生成深度和幅度的数值向量
    encoder.continuous_mapping(min_val=0, max_val=300, num_levels=301, base_key='depths')
    encoder.continuous_mapping(min_val=0, max_val=100, num_levels=101, base_key='amplitudes')
    
    # 3. 准备数据
    processor = BScanDataProcessor(encoder)
    samples, labels = processor.create_training_data(num_samples=500)
    
    # 4. 训练分类器
    classifier = HDCClassifier(encoder)
    classifier.train(samples, labels)
    
    # 5. 测试
    test_sample = (1, 0, 125, 80)  # 通道1, 深度125mm, 幅度80%
    predicted_label, similarities = classifier.predict(test_sample)
    
    print(f"测试样本预测结果: {predicted_label}")
    print(f"各类别相似度: {similarities}")
    
    # 6. 重训练优化
    print("\n开始重训练优化...")
    classifier.retrain(samples, labels, iterations=5)
    
    # 7. 重训练后重新测试
    predicted_label, similarities = classifier.predict(test_sample)
    print(f"\n重训练后预测结果: {predicted_label}")
    print(f"各类别相似度: {similarities}")


if __name__ == "__main__":
    main()