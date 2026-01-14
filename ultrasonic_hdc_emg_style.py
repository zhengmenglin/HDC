import torch
import torch.nn as nn
import torch.utils.data as data
import torchmetrics
from tqdm import tqdm
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from pathlib import Path
import pandas as pd
import numpy as np

# ======================================
# 设备设置
# ======================================
# 自动选择设备，优先使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================================
# 配置参数
# ======================================
# 超向量维度，HDC模型的核心参数
DIMENSIONS = 10000
# 水平嵌入的级别数，用于将连续信号离散化为有限级别
NUM_LEVELS = 21
# 批处理大小，每次训练使用的样本数
BATCH_SIZE = 1
# 窗口大小，每个样本包含的编码值数量
WINDOW = 256
# 通道数，超声波传感器的通道数量
CHANNELS = 9
# 降采样因子，减少数据维度
DOWNSAMPLE = 5
# 降采样索引，用于从窗口中选择采样点
SUBSAMPLES = torch.arange(0, WINDOW, int(WINDOW / DOWNSAMPLE))
# N-gram大小，用于捕获序列数据的时序关系
N_GRAM_SIZE = 4

# 数据文件路径
csv_file = Path(r"d:\HDC\data\train\接头3.cap_hdc.csv")


# ======================================
# 数据转换函数
# ======================================
def transform(x):
    """降采样转换，减少数据维度
    
    Args:
        x: 输入数据，形状为 [window_size, channels]
        
    Returns:
        降采样后的数据，形状为 [downsample_size, channels]
    """
    return x[SUBSAMPLES]


# ======================================
# 超声波数据集类
# ======================================
class UltrasonicDataset(data.Dataset):
    """超声波数据集类，用于加载和处理超声波数据
    
    Args:
        csv_file: CSV文件路径
        indices: 样本索引列表
        transform: 数据转换函数
    """
    def __init__(self, csv_file, indices, transform=None):
        self.csv_file = csv_file
        self.indices = indices
        self.transform = transform
        self.samples = []  # 存储所有样本
        self.labels = []   # 存储对应标签
        
        # 加载所有数据
        self._load_all_data()
    
    def _load_all_data(self):
        """加载所有数据并生成滑动窗口样本"""
        print(f"正在加载数据文件: {self.csv_file}")
        
        # 读取CSV数据
        df = pd.read_csv(self.csv_file)
        
        # 按编码值排序，确保数据的时序连续性
        df = df.sort_values(by='encoding')
        
        # 提取通道数据和标签
        channel_data = df[[f'channel_{i}' for i in range(CHANNELS)]].values
        labels = df['label'].values
        encodings = df['encoding'].values
        
        print(f"从 {self.csv_file} 加载了 {len(encodings)} 个编码值")
        
        # 确保有足够的数据生成至少一个窗口
        if len(encodings) < WINDOW:
            print(f"警告: 数据不足，无法生成窗口。需要 {WINDOW} 个编码值，仅获得 {len(encodings)} 个")
            return
        
        # 生成所有可能的滑动窗口样本
        all_samples = []
        all_labels = []
        
        # 滑动窗口生成循环
        for start in range(0, len(encodings) - WINDOW + 1, WINDOW):
            end = start + WINDOW
            
            # 提取窗口内的通道数据，形状为 [window_size, channels]
            window_data = channel_data[start:end]
            
            # 转换为float32格式
            sample = window_data.astype(np.float32)
            
            # 确定窗口标签，使用窗口内出现最多的标签
            window_labels = labels[start:end]
            if len(window_labels) > 0:
                label = max(set(window_labels), key=list(window_labels).count)
            else:
                label = 0
            
            all_samples.append(sample)
            all_labels.append(label)
        
        print(f"共生成 {len(all_samples)} 个窗口样本")
        if all_labels:
            print(f"标签分布: {np.bincount(all_labels)}")
        
        # 根据索引选择样本，用于划分训练集和测试集
        self.samples = [all_samples[i] for i in self.indices]
        self.labels = [all_labels[i] for i in self.indices]
        
        print(f"当前数据集包含 {len(self.samples)} 个样本")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本和标签
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 样本数据
            label: 样本标签
        """
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量
        sample = torch.from_numpy(sample)
        
        # 应用数据转换
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# ======================================
# 超维编码器类
# ======================================
class Encoder(nn.Module):
    """超维编码器，将时序信号转换为超维向量
    
    Args:
        out_features: 输出超维向量的维度
        timestamps: 输入时序数据的时间戳数量
        channels: 输入数据的通道数量
    """
    def __init__(self, out_features, timestamps, channels):
        super(Encoder, self).__init__()
        
        # 通道嵌入层，将每个通道映射到超维空间
        self.channels = embeddings.Random(channels, out_features)
        # 信号嵌入层，将连续信号值映射到超维空间
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=20)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播，将输入转换为超维向量
        
        Args:
            input: 输入数据，形状为 [batch, timestamps, channels]
            
        Returns:
            超维向量，形状为 [batch, out_features]
        """
        # 将连续信号嵌入到超维空间
        signal = self.signals(input)
        # 将通道信息与信号嵌入结合
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        
        # 使用multiset聚合样本
        samples = torchhd.multiset(samples)
        # 使用n-gram捕获时序关系
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)
        # 硬量化，将超维向量转换为二进制向量
        return torchhd.hard_quantize(sample_hv)


# ======================================
# 实验函数
# ======================================
def experiment():
    """训练和评估HDC模型"""
    print("开始超声波HDC实验...")
    
    # 计算总共有多少个窗口样本
    print("正在生成样本...")
    
    # 先读取数据计算样本数量
    df = pd.read_csv(csv_file)
    encodings = df['encoding'].values
    total_samples = len(range(0, len(encodings) - WINDOW + 1, WINDOW))
    print(f"总共有 {total_samples} 个可能的窗口样本")
    
    # 创建索引并随机分配到训练和测试集
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    
    # 70%训练，30%测试
    train_size = int(total_samples * 0.7)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 加载数据集
    print(f"创建数据集：{len(train_indices)} 个训练样本，{len(test_indices)} 个测试样本")
    train_ds = UltrasonicDataset(csv_file, train_indices, transform=transform)
    test_ds = UltrasonicDataset(csv_file, test_indices, transform=transform)
    
    # 获取所有唯一标签，确保模型包含所有可能的类别
    all_labels = []
    if train_ds.labels:
        all_labels.extend(train_ds.labels)
    if test_ds.labels:
        all_labels.extend(test_ds.labels)
    
    # 计算实际类别数量
    if all_labels:
        max_label = max(all_labels)
        num_classes = int(max_label + 1)  # 转换为Python原生int类型
        print(f"检测到 {num_classes} 个唯一类别（标签 0 到 {max_label}）")
    else:
        num_classes = 1
        print("警告: 未找到标签，使用 1 个类别")
    
    # 创建数据加载器
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 检查数据集是否为空
    if len(train_ds) == 0:
        print("错误: 训练数据集为空！")
        return
    
    # 初始化编码器
    # 样本形状: [DOWNSAMPLE, channels]
    sample_shape = train_ds[0][0].shape
    encode = Encoder(DIMENSIONS, sample_shape[0], sample_shape[1])
    encode = encode.to(device)
    
    # 初始化模型，使用Centroid（质心）模型
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    
    # 训练模型
    print("正在训练模型...")
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="训练中"):
            samples = samples.to(device)
            targets = targets.to(device)
            
            # 生成超维向量
            sample_hv = encode(samples)
            # 更新模型质心
            model.add(sample_hv, targets)
    
    # 评估模型
    print("正在评估模型...")
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    
    with torch.no_grad():
        # 归一化模型
        model.normalize()
        
        # 测试模型
        for samples, targets in tqdm(test_ld, desc="测试中"):
            samples = samples.to(device)
            
            # 生成超维向量
            sample_hv = encode(samples)
            # 模型预测
            output = model(sample_hv, dot=True)
            # 更新准确率指标
            accuracy.update(output.cpu(), targets)
    
    # 打印测试准确率
    print(f"测试准确率: {(accuracy.compute().item() * 100):.3f}%")


if __name__ == "__main__":
    # 运行实验
    experiment()