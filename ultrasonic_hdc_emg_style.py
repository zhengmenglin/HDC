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

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 配置参数
DIMENSIONS = 10000  # 超向量维度
NUM_LEVELS = 21     # 水平嵌入的级别数
BATCH_SIZE = 1      # 批大小
WINDOW = 64         # 窗口大小（编码数）
CHANNELS = 9        # 通道数
FEATURES = 1        # 每个通道的特征数（只有深度值）
EFFECTIVE_CHANNELS = CHANNELS * FEATURES  # 有效通道数 = 9*1=9
DOWNSAMPLE = 8      # 降采样因子
SUBSAMPLES = torch.arange(0, WINDOW, int(WINDOW / DOWNSAMPLE))  # 降采样索引
N_GRAM_SIZE = 4     # N-gram大小
STEP = 32           # 滑动窗口步长

train_data_dir = Path(r"d:\HDC\data\train")
test_data_dir = Path(r"d:\HDC\data\test")


def transform(x):
    """降采样转换，减少数据维度"""
    return x[SUBSAMPLES]


class UltrasonicDataset(data.Dataset):
    """超声波数据集"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # 加载所有CSV文件
        self._load_data()
    
    def _load_data(self):
        """加载数据并生成滑动窗口"""
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        
        for csv_file in csv_files:
            print(f"Loading {csv_file}")
            df = pd.read_csv(csv_file)
            
            # 检测CSV格式
            has_encoding = 'encoding' in df.columns
            has_channel = 'channel_0' in df.columns
            
            if has_encoding and has_channel:
                 # 旧格式: encoding, channel_0, channel_1, ..., label
                 print("Detected old format (encoding, channel_*, label)")
                 # 按编码值排序
                 df = df.sort_values(by='encoding')
                 
                 # 提取通道数据和标签
                 channel_data = df[[f'channel_{i}' for i in range(CHANNELS)]].values
                 labels = df['label'].values
                 encodings = df['encoding'].values
                 
                 print(f"Loaded {len(encodings)} E values from {csv_file}")
                 
                 # 生成滑动窗口
                 encoding_count = len(encodings)
                 
                 for start_idx in range(0, encoding_count - WINDOW + 1, STEP):
                     end_idx = start_idx + WINDOW
                     
                     # 提取窗口内的通道数据 [window_size, channels]
                     window_channel_data = channel_data[start_idx:end_idx]
                     
                     # 重塑为 [channels, window_size, 1] 然后扩展为 [channels, window_size, 3]
                     sample = window_channel_data.T.reshape(CHANNELS, WINDOW, 1)
                     sample = np.repeat(sample, FEATURES, axis=2)  # 扩展为3个特征
                     
                     # 确定窗口标签（使用窗口内出现最多的标签）
                     window_labels = labels[start_idx:end_idx]
                     if len(window_labels) > 0:
                         label = max(set(window_labels), key=list(window_labels).count)
                     else:
                         label = 0
                     
                     # 转换为PyTorch张量
                     sample = torch.from_numpy(sample)
                     # 转置为 [window_size, channels, features] 以便处理
                     sample = sample.permute(1, 0, 2)
                     # 合并特征到通道维度 [window_size, channels*features]
                     sample = sample.reshape(WINDOW, -1)
                     # 转换回NumPy数组存储
                     sample = sample.numpy()
                     
                     self.samples.append(sample)
                     self.labels.append(label)
            
            elif has_channel == False:
                 # 新格式: feature_0, feature_1, ..., label
                 # 每行已经是一个展平的窗口数据
                 print("Detected new format (feature_*, label) - each row is a flattened window")
                 
                 for idx, row in df.iterrows():
                     # 提取特征和标签
                     features = row.values[:-1]
                     label = int(row.values[-1])
                     
                     # 新格式每行已经是一个展平的窗口 [1728] = [9*64*3]
                     # 直接使用，不需要重塑
                     sample = np.array(features, dtype=np.float32)
                     
                     self.samples.append(sample)
                     self.labels.append(label)
            else:
                print(f"Unknown format in {csv_file}, skipping")
                continue
        
        print(f"Total samples loaded: {len(self.samples)}")
        if self.labels:
            print(f"Label distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


class Encoder(nn.Module):
    """超维编码器"""
    def __init__(self, out_features, timestamps, channels):
        super(Encoder, self).__init__()
        
        self.channels = embeddings.Random(channels, out_features)
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=20)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x形状: [batch, timestamps, channels]
        signal = self.signals(x)  # [batch, timestamps, channels, out_features]
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))  # [batch, timestamps, channels, out_features]
        
        samples = torchhd.multiset(samples)  # [batch, channels, out_features]
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)  # [batch, out_features]
        return torchhd.hard_quantize(sample_hv)


def experiment():
    """训练和评估模型"""
    print("Starting ultrasonic HDC experiment...")
    
    # 加载数据集
    print("Loading training dataset...")
    train_ds = UltrasonicDataset(train_data_dir, transform=transform)
    
    print("Loading test dataset...")
    test_ds = UltrasonicDataset(test_data_dir, transform=transform)
    
    # 获取所有唯一标签，确保模型包含所有可能的类别
    all_labels = []
    if train_ds.labels:
        all_labels.extend(train_ds.labels)
    if test_ds.labels:
        all_labels.extend(test_ds.labels)
    
    # 计算实际类别数量（最大标签值 + 1，确保包含所有类别）
    if all_labels:
        max_label = max(all_labels)
        num_classes = int(max_label + 1)  # 转换为Python原生int类型
        print(f"Detected {num_classes} unique classes (labels 0 to {max_label})")
    else:
        num_classes = 1
        print("Warning: No labels found, using 1 class")
    
    # 创建数据加载器
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 检查数据集是否为空
    if len(train_ds) == 0:
        print("Error: Training dataset is empty!")
        return
    
    # 初始化编码器
    # 样本形状: [DOWNSAMPLE, EFFECTIVE_CHANNELS]
    sample_shape = train_ds[0][0].shape
    encode = Encoder(DIMENSIONS, sample_shape[0], sample_shape[1])
    encode = encode.to(device)
    
    # 初始化模型
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    
    # 训练模型
    print("Training model...")
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            targets = targets.to(device)
            
            sample_hv = encode(samples)
            model.add(sample_hv, targets)
    
    # 评估模型
    print("Evaluating model...")
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    
    with torch.no_grad():
        model.normalize()
        
        for samples, targets in tqdm(test_ld, desc="Testing"):
            samples = samples.to(device)
            
            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            accuracy.update(output.cpu(), targets)
    
    print(f"Testing accuracy: {(accuracy.compute().item() * 100):.3f}%")


if __name__ == "__main__":
    experiment()