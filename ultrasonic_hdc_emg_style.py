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
import os

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
WINDOW = 128
# 滑动窗口步长
STEP = 64
# 通道数，超声波传感器的通道数量
CHANNELS = 9
# N-gram大小，用于捕获序列数据的时序关系
N_GRAM_SIZE = 4

# 数据目录
TRAIN_DIR = Path(r"d:\HDC\data\train")
TEST_DIR = Path(r"d:\HDC\data\test")
# 模型保存路径
MODEL_PATH = Path(r"d:\HDC\models\hdc_model.pth")
# 确保模型目录存在
MODEL_PATH.parent.mkdir(exist_ok=True)





# ======================================
# 超声波数据集类
# ======================================
class UltrasonicDataset(data.Dataset):
    """超声波数据集类，用于加载和处理超声波数据
    
    Args:
        data_dir: 数据目录路径
        is_train: 是否为训练模式
    """
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []  # 存储所有样本
        self.labels = []   # 存储对应标签
        self.file_info = []  # 存储每个样本对应的文件名和索引信息
        
        # 加载所有数据
        self._load_all_data()
    
    def _load_all_data(self):
        """加载所有数据并生成样本"""
        print(f"正在加载 {self.data_dir} 目录下的所有CSV文件")
        
        # 获取所有CSV文件
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            print(f"\n正在处理文件: {csv_file.name}")
            
            # 读取CSV数据
            df = pd.read_csv(csv_file)
            
            # 按编码值排序，确保数据的时序连续性
            df = df.sort_values(by='encoding')
            
            # 提取通道数据和标签
            channel_data = df[[f'channel_{i}' for i in range(CHANNELS)]].values
            labels = df['label'].values if 'label' in df.columns else [0] * len(df)
            encodings = df['encoding'].values
            
            print(f"从 {csv_file.name} 加载了 {len(encodings)} 个编码值")
            
            # 确保有足够的数据生成至少一个窗口
            if len(encodings) < WINDOW:
                print(f"警告: 数据不足，无法生成窗口。需要 {WINDOW} 个编码值，仅获得 {len(encodings)} 个")
                continue
            
            # 生成所有可能的滑动窗口样本
            file_samples = []
            file_labels = []
            file_indices = []
            
            # 滑动窗口生成循环
            for start in range(0, len(encodings) - WINDOW + 1, STEP):
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
                
                file_samples.append(sample)
                file_labels.append(label)
                # 保存文件信息和窗口索引
                file_indices.append((str(csv_file), start, end))
            
            # 将当前文件的样本添加到总样本列表
            self.samples.extend(file_samples)
            self.labels.extend(file_labels)
            self.file_info.extend(file_indices)
            
            print(f"从 {csv_file.name} 生成了 {len(file_samples)} 个窗口样本")
        
        print(f"\n所有文件处理完成，共生成 {len(self.samples)} 个窗口样本")
        if self.labels:
            print(f"标签分布: {np.bincount(self.labels)}")
    
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
        n_gram_size: n-gram大小，用于捕获时序关系
    """
    def __init__(self, out_features, timestamps, channels, n_gram_size=N_GRAM_SIZE):
        super(Encoder, self).__init__()
        
        # 通道嵌入层，将每个通道映射到超维空间
        self.channels = embeddings.Random(channels, out_features)
        # 信号嵌入层，将连续信号值映射到超维空间
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=20)
        # n-gram大小
        self.n_gram_size = n_gram_size
    
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
        sample_hv = torchhd.ngrams(samples, n=self.n_gram_size)
        # 硬量化，将超维向量转换为二进制向量
        return torchhd.hard_quantize(sample_hv)


# ======================================
# 训练函数
# ======================================
def train_model(n_gram_size=N_GRAM_SIZE):
    """使用训练数据训练HDC模型
    
    Args:
        n_gram_size: N-gram大小，用于捕获时序关系
    """
    print(f"开始模型训练... (N_GRAM_SIZE: {n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP})")
    
    # 加载训练数据集
    train_ds = UltrasonicDataset(TRAIN_DIR, is_train=True)
    
    # 创建数据加载器
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 检查数据集是否为空
    if len(train_ds) == 0:
        print("错误: 训练数据集为空！")
        return False
    
    # 获取所有唯一标签，确保模型包含所有可能的类别
    all_labels = train_ds.labels
    if all_labels:
        max_label = max(all_labels)
        num_classes = int(max_label + 1)  # 转换为Python原生int类型
        print(f"检测到 {num_classes} 个唯一类别（标签 0 到 {max_label}）")
    else:
        num_classes = 1
        print("警告: 未找到标签，使用 1 个类别")
    
    # 初始化编码器
    # 样本形状: [downsample_factor, channels]
    sample_shape = train_ds[0][0].shape
    encode = Encoder(DIMENSIONS, sample_shape[0], sample_shape[1], n_gram_size=n_gram_size)
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
    
    # 归一化模型
    model.normalize()
    
    # 保存模型和编码器
    torch.save({
        'encoder_state_dict': encode.state_dict(),
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'sample_shape': sample_shape,
        'n_gram_size': n_gram_size
    }, MODEL_PATH)
    
    print(f"模型训练完成，已保存到: {MODEL_PATH}")
    return True


# ======================================
# 预测函数
# ======================================
def predict_model(data_dir=TEST_DIR):
    """使用训练好的模型对数据进行预测
    
    Args:
        data_dir: 预测数据目录，默认使用测试数据目录
    """
    print("开始模型预测...")
    
    # 检查模型文件是否存在
    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return False
    
    # 加载模型和编码器
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    
    # 获取模型训练时的N_GRAM_SIZE
    n_gram_size = checkpoint.get('n_gram_size', N_GRAM_SIZE)
    print(f"预测时使用模型训练参数: N_GRAM_SIZE={n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP}")
    
    # 初始化编码器和模型
    sample_shape = checkpoint['sample_shape']
    encode = Encoder(DIMENSIONS, sample_shape[0], sample_shape[1], n_gram_size=n_gram_size)
    encode.load_state_dict(checkpoint['encoder_state_dict'])
    encode = encode.to(device)
    encode.eval()
    
    num_classes = checkpoint['num_classes']
    model = Centroid(DIMENSIONS, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载数据集，使用与训练时相同的窗口大小和滑动步长
    test_ds = UltrasonicDataset(data_dir, is_train=False)
    
    # 创建数据加载器
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 检查数据集是否为空
    if len(test_ds) == 0:
        print(f"错误: {data_dir} 目录下的数据集为空！")
        return False
    
    # 执行预测
    print(f"正在对 {data_dir} 目录下的数据进行预测...")
    predictions = []
    with torch.no_grad():
        for samples, targets in tqdm(test_ld, desc="预测中"):
            samples = samples.to(device)
            
            # 生成超维向量
            sample_hv = encode(samples)
            # 模型预测
            output = model(sample_hv, dot=True)
            # 获取预测标签
            pred_label = torch.argmax(output, dim=1).cpu().item()
            predictions.append(pred_label)
    
    # 保存预测结果
    print("正在保存预测结果...")
    
    # 按文件分组预测结果
    file_predictions = {}
    for idx, pred_label in enumerate(predictions):
        file_path, start, end = test_ds.file_info[idx]
        file_key = Path(file_path).name
        if file_key not in file_predictions:
            file_predictions[file_key] = []
        file_predictions[file_key].append((start, end, pred_label))
    
    # 为每个文件生成预测结果CSV
    for file_name, pred_info in file_predictions.items():
        # 读取原始文件
        original_file = data_dir / file_name
        df = pd.read_csv(original_file)
        
        # 创建预测结果列
        df['predicted_label'] = df['label'] if 'label' in df.columns else 0
        
        # 根据预测结果更新标签
        for start, end, pred_label in pred_info:
            df.loc[start:end-1, 'predicted_label'] = pred_label
        
        # 保存带预测结果的CSV文件，确保只添加一个hdc后缀
        output_file = data_dir / f"{original_file.stem.replace('_hdc', '')}_hdc.csv"
        df.to_csv(output_file, index=False)
        print(f"已保存预测结果到: {output_file}")
    
    print("所有预测结果已保存")
    return True


# ======================================
# 主函数
# ======================================
def main():
    """训练模型并进行预测"""
    # 训练模型
    if not train_model():
        return
    
    # 进行预测
    if not predict_model():
        return
    
    print("\n=== 训练和预测完成 ===")


if __name__ == "__main__":
    # 运行主函数
    main()