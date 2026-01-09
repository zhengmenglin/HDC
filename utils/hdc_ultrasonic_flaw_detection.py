import numpy as np
import random
import matplotlib
# 使用非交互式后端，适合命令行环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks

# 超维计算参数
DIMENSION = 10000  # 超向量维度
SEED = 42  # 随机种子

# 设置随机种子，保证结果可复现
np.random.seed(SEED)
random.seed(SEED)

class HyperdimensionalComputing:
    def __init__(self, dimension=DIMENSION):
        self.dimension = dimension
        
    def generate_random_vector(self):
        """生成随机超向量，元素为±1"""
        return np.random.choice([-1, 1], size=self.dimension)
    
    def bind(self, vec1, vec2):
        """绑定操作：逐元素乘法"""
        return np.multiply(vec1, vec2)
    
    def permute(self, vec, shift=1):
        """置换操作：循环移位"""
        return np.roll(vec, shift)
    
    def bundle(self, vec_list):
        """捆绑操作：向量加法"""
        result = np.zeros(self.dimension)
        for vec in vec_list:
            result += vec
        # 将结果二值化，大于0为1，否则为-1
        result[result > 0] = 1
        result[result <= 0] = -1
        return result
    
    def similarity(self, vec1, vec2):
        """计算向量相似性：点积"""
        return np.dot(vec1, vec2) / self.dimension

class UltrasonicSignalGenerator:
    def __init__(self):
        pass
    
    def generate_normal_signal(self, duration=0.1, fs=100000):
        """生成正常的超声波信号"""
        t = np.linspace(0, duration, int(duration * fs))
        # 生成一个chirp信号模拟超声波
        signal = chirp(t, f0=10000, f1=50000, t1=duration, method='linear')
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, len(signal))
        return signal + noise
    
    def generate_defect_signal(self, duration=0.1, fs=100000):
        """生成包含缺陷的超声波信号"""
        t = np.linspace(0, duration, int(duration * fs))
        # 生成基本chirp信号
        signal = chirp(t, f0=10000, f1=50000, t1=duration, method='linear')
        # 在信号中间添加一个缺陷回波
        defect_time = duration / 2
        defect_amplitude = 0.5
        defect_width = 0.001
        defect = defect_amplitude * np.exp(-((t - defect_time) / defect_width)**2)
        # 添加噪声
        noise = np.random.normal(0, 0.1, len(signal))
        return signal + defect + noise

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, signal, fs=100000):
        """从超声波信号中提取特征"""
        # 时域特征
        peak_amplitude = np.max(np.abs(signal))
        rms = np.sqrt(np.mean(signal**2))
        # 频域特征
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), 1/fs)
        positive_freq = freq[:len(freq)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        # 找到频谱峰值
        peaks, _ = find_peaks(positive_fft)
        dominant_freq = positive_freq[peaks[np.argmax(positive_fft[peaks])]] if len(peaks) > 0 else 0
        
        return {
            'peak_amplitude': peak_amplitude,
            'rms': rms,
            'dominant_freq': dominant_freq
        }

class HDCSonicFlawDetector:
    def __init__(self, dimension=DIMENSION):
        self.hdc = HyperdimensionalComputing(dimension)
        self.signal_gen = UltrasonicSignalGenerator()
        self.feature_extractor = FeatureExtractor()
        
        # 创建特征基向量
        self.feature_bases = {
            'peak_amplitude': self.hdc.generate_random_vector(),
            'rms': self.hdc.generate_random_vector(),
            'dominant_freq': self.hdc.generate_random_vector()
        }
        
        # 类别原型向量（初始为空）
        self.normal_prototype = None
        self.defect_prototype = None
    
    def encode_features(self, features):
        """将提取的特征编码为超向量"""
        encoded_vectors = []
        
        for feature_name, feature_value in features.items():
            if feature_name in self.feature_bases:
                # 特征值量化
                quantized_value = 1 if feature_value > 0.5 else -1
                # 绑定特征基向量和量化值
                encoded_vec = self.hdc.bind(self.feature_bases[feature_name], np.full(self.hdc.dimension, quantized_value))
                encoded_vectors.append(encoded_vec)
        
        # 捆绑所有编码向量
        return self.hdc.bundle(encoded_vectors)
    
    def train(self, num_samples=50):
        """训练模型，生成类别原型向量"""
        normal_vectors = []
        defect_vectors = []
        
        # 生成正常样本
        for _ in range(num_samples):
            signal = self.signal_gen.generate_normal_signal()
            features = self.feature_extractor.extract_features(signal)
            encoded_vec = self.encode_features(features)
            normal_vectors.append(encoded_vec)
        
        # 生成缺陷样本
        for _ in range(num_samples):
            signal = self.signal_gen.generate_defect_signal()
            features = self.feature_extractor.extract_features(signal)
            encoded_vec = self.encode_features(features)
            defect_vectors.append(encoded_vec)
        
        # 生成类别原型向量
        self.normal_prototype = self.hdc.bundle(normal_vectors)
        self.defect_prototype = self.hdc.bundle(defect_vectors)
        
        print(f"训练完成，生成了{num_samples}个正常样本和{num_samples}个缺陷样本的原型向量")
    
    def classify(self, signal):
        """对超声波信号进行分类"""
        if self.normal_prototype is None or self.defect_prototype is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 提取特征
        features = self.feature_extractor.extract_features(signal)
        # 编码为超向量
        encoded_vec = self.encode_features(features)
        
        # 计算与原型向量的相似性
        sim_normal = self.hdc.similarity(encoded_vec, self.normal_prototype)
        sim_defect = self.hdc.similarity(encoded_vec, self.defect_prototype)
        
        # 分类
        if sim_defect > sim_normal:
            return "defect", sim_defect, sim_normal
        else:
            return "normal", sim_defect, sim_normal
    
    def test(self, num_test_samples=20):
        """测试模型性能"""
        correct = 0
        total = 0
        
        # 测试正常样本
        for _ in range(num_test_samples):
            signal = self.signal_gen.generate_normal_signal()
            result, _, _ = self.classify(signal)
            if result == "normal":
                correct += 1
            total += 1
        
        # 测试缺陷样本
        for _ in range(num_test_samples):
            signal = self.signal_gen.generate_defect_signal()
            result, _, _ = self.classify(signal)
            if result == "defect":
                correct += 1
            total += 1
        
        accuracy = correct / total
        print(f"测试完成，准确率：{accuracy:.2f} ({correct}/{total})")
        return accuracy
    
    def visualize_signal(self, signal, title="超声波信号", fs=100000, save_path=None):
        """可视化超声波信号波形和频谱"""
        t = np.linspace(0, len(signal)/fs, len(signal))
        
        # 计算FFT
        fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), 1/fs)
        positive_freq = freq[:len(freq)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        # 创建画布
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制时域波形
        ax1.plot(t, signal)
        ax1.set_title(f"{title} - 时域波形")
        ax1.set_xlabel("时间 (s)")
        ax1.set_ylabel("振幅")
        ax1.grid(True)
        
        # 绘制频域频谱
        ax2.plot(positive_freq / 1000, positive_fft)
        ax2.set_title(f"{title} - 频域频谱")
        ax2.set_xlabel("频率 (kHz)")
        ax2.set_ylabel("幅度")
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图形到文件
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")
        plt.close()
    
    def visualize_classification(self, normal_signal, defect_signal, fs=100000):
        """可视化正常样本和缺陷样本的分类结果"""
        # 分类正常样本
        normal_result, normal_sim_defect, normal_sim_normal = self.classify(normal_signal)
        
        # 分类缺陷样本
        defect_result, defect_sim_defect, defect_sim_normal = self.classify(defect_signal)
        
        # 可视化信号并保存为文件
        self.visualize_signal(normal_signal, title="正常样本超声波信号", 
                             save_path="normal_signal.png")
        self.visualize_signal(defect_signal, title="缺陷样本超声波信号", 
                             save_path="defect_signal.png")
        
        # 绘制分类结果对比图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 数据准备
        categories = ['正常样本', '缺陷样本']
        defect_similarities = [normal_sim_defect, defect_sim_defect]
        normal_similarities = [normal_sim_normal, defect_sim_normal]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # 绘制柱状图
        ax.bar(x - width/2, defect_similarities, width, label='缺陷相似度')
        ax.bar(x + width/2, normal_similarities, width, label='正常相似度')
        
        # 设置标题和标签
        ax.set_title('超维计算分类结果对比')
        ax.set_xlabel('样本类型')
        ax.set_ylabel('相似度')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # 添加数值标签
        for i, v in enumerate(defect_similarities):
            ax.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
        for i, v in enumerate(normal_similarities):
            ax.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
        
        # 添加分类结果文本
        ax.text(0, -0.1, f'分类结果: {normal_result}', ha='center', transform=ax.transAxes)
        ax.text(1, -0.1, f'分类结果: {defect_result}', ha='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # 保存分类结果图
        plt.savefig("classification_result.png", dpi=300, bbox_inches='tight')
        print("分类结果图已保存到: classification_result.png")
        plt.close()

# 主程序示例
if __name__ == "__main__":
    # 创建探伤器实例
    detector = HDCSonicFlawDetector()
    
    # 训练模型
    detector.train(num_samples=50)
    
    # 测试模型
    detector.test(num_test_samples=20)
    
    # 单个样本测试示例
    print("\n单个样本测试：")
    
    # 生成测试样本
    normal_signal = detector.signal_gen.generate_normal_signal()
    defect_signal = detector.signal_gen.generate_defect_signal()
    
    # 测试正常样本
    result, sim_defect, sim_normal = detector.classify(normal_signal)
    print(f"正常样本分类结果：{result} (缺陷相似度：{sim_defect:.4f}, 正常相似度：{sim_normal:.4f})")
    
    # 测试缺陷样本
    result, sim_defect, sim_normal = detector.classify(defect_signal)
    print(f"缺陷样本分类结果：{result} (缺陷相似度：{sim_defect:.4f}, 正常相似度：{sim_normal:.4f})")
    
    # 图形化显示
    print("\n正在生成可视化结果...")
    detector.visualize_classification(normal_signal, defect_signal)