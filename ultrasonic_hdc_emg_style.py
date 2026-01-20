import sys
import os
# 添加torchhd库的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

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
# 超向量维度，HDC模型的核心参数（根据理论建议设为10000）
DIMENSIONS = 10000
# 水平嵌入的级别数，用于将连续信号离散化为有限级别
NUM_LEVELS = 1000  # 增加级别数以提高精度
# 批处理大小，每次训练使用的样本数
BATCH_SIZE = 1
# 窗口大小，每个样本包含的编码值数量
# 根据文档分析，平均11行构成一个伤损或结构，适当增加以捕获更多上下文
WINDOW = 24  # 优化窗口大小
# 滑动窗口步长，较小的步长可以提供更多训练样本
STEP = 6  # 减小步长以增加样本数量
# 通道数，超声波传感器的通道数量
CHANNELS = 9
# N-gram大小，用于捕获序列数据的时序关系
N_GRAM_SIZE = 3  # 减小以降低计算复杂度

# 数据目录
TRAIN_DIR = Path(r"d:\HDC\data\train")
TEST_DIR = Path(r"d:\HDC\data\test")
# 模型保存路径
MODEL_PATH = Path(r"d:\HDC\models\hdc_model.pth")
# 确保模型目录存在
MODEL_PATH.parent.mkdir(exist_ok=True)





# ======================================
# 数据预处理函数
# ======================================
def preprocess_ultrasonic_data(data: np.ndarray, normalize: bool = True, denoise: bool = False) -> np.ndarray:
    """超声波数据预处理

    Args:
        data: 输入数据，形状为 [timestamps, channels]
        normalize: 是否进行归一化
        denoise: 是否进行去噪处理

    Returns:
        处理后的数据
    """
    processed_data = data.copy()

    if denoise:
        # 简单的滑动平均去噪
        kernel_size = 3
        for channel in range(data.shape[1]):
            # 使用一维卷积进行滑动平均
            kernel = np.ones(kernel_size) / kernel_size
            processed_data[:, channel] = np.convolve(data[:, channel], kernel, mode='same')

    if normalize:
        # 对每个通道分别进行归一化
        for channel in range(data.shape[1]):
            channel_data = processed_data[:, channel]
            # 使用min-max归一化到[0,1]范围
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            if max_val > min_val:
                processed_data[:, channel] = (channel_data - min_val) / (max_val - min_val)
            else:
                # 如果所有值相同，设为0.5
                processed_data[:, channel] = 0.5

    return processed_data


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
            
            # 检查是否有多通道标签
            if all([f'label_{i}' in df.columns for i in range(CHANNELS)]):
                # 使用多通道标签
                labels = df[[f'label_{i}' for i in range(CHANNELS)]].values
                print(f"使用多通道标签，形状: {labels.shape}")
            else:
                # 使用单通道标签（向后兼容）
                labels = df['label'].values if 'label' in df.columns else [0] * len(df)
                # 扩展为多通道标签格式
                labels = np.tile(labels[:, np.newaxis], (1, CHANNELS))
                print(f"使用单通道标签并扩展为多通道格式，形状: {labels.shape}")
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

                # 数据预处理：归一化和去噪
                window_data = preprocess_ultrasonic_data(window_data, normalize=True, denoise=False)

                # 转换为float32格式
                sample = window_data.astype(np.float32)
                
                # 确定窗口标签，使用窗口内出现最多的标签
                window_labels = labels[start:end]
                if len(window_labels) > 0:
                    if window_labels.ndim == 2:  # 多通道标签
                        # 对每个通道分别计算出现最多的标签
                        label = []
                        for channel in range(window_labels.shape[1]):
                            channel_labels = window_labels[:, channel]
                            channel_label = max(set(channel_labels), key=list(channel_labels).count)
                            label.append(channel_label)
                        label = np.array(label)
                    else:  # 单通道标签
                        label = max(set(window_labels), key=list(window_labels).count)
                else:
                    label = np.zeros(CHANNELS) if labels.ndim == 2 else 0
                
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
            if isinstance(self.labels[0], np.ndarray):
                # 多通道标签，分别打印每个通道的分布
                for channel in range(self.labels[0].shape[0]):
                    channel_labels = [label[channel] for label in self.labels]
                    print(f"通道 {channel} 标签分布: {np.bincount(channel_labels)}")
            else:
                # 单通道标签
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

    根据超维计算理论，实现：
    1. 通道分离处理：9个通道分开处理
    2. 循环移位编码：表示多个点的深度值
    3. 特征融合：将多个通道的超向量融合

    Args:
        out_features: 输出超维向量的维度
        timestamps: 输入时序数据的时间戳数量
        channels: 输入数据的通道数量
        n_gram_size: n-gram大小，用于捕获时序关系
    """
    def __init__(self, out_features, timestamps, channels, n_gram_size=N_GRAM_SIZE):
        super(Encoder, self).__init__()

        # 每个通道独立的嵌入层（通道分离处理）
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                # 信号嵌入层，将连续信号值映射到超维空间
                embeddings.Level(NUM_LEVELS, out_features, high=20),
                # 位置嵌入层，用于循环移位编码
                embeddings.Random(timestamps, out_features)
            ) for _ in range(channels)
        ])

        # n-gram大小
        self.n_gram_size = n_gram_size
        self.channels = channels
        self.out_features = out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播，将输入转换为超维向量

        Args:
            input: 输入数据，形状为 [batch, timestamps, channels]

        Returns:
            超维向量，形状为 [batch, out_features]
        """
        batch_size = input.shape[0]

        # 通道分离处理：每个通道独立编码
        channel_hvs = []
        for channel_idx in range(self.channels):
            # 提取当前通道的数据 [batch, timestamps]
            channel_data = input[:, :, channel_idx]

            # 获取当前通道的编码器
            signal_embed, position_embed = self.channel_encoders[channel_idx]

            # 信号嵌入：将深度值映射到超维空间
            signal_hv = signal_embed(channel_data)  # [batch, timestamps, out_features]

            # 位置嵌入：用于表示时间位置（循环移位编码的基础）
            position_hv = position_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, timestamps, out_features]

            # 绑定信号和位置信息
            channel_hv = torchhd.bind(signal_hv, position_hv)  # [batch, timestamps, out_features]

            # 对时间维度进行multiset聚合
            channel_hv = torchhd.multiset(channel_hv)  # [batch, out_features]

            channel_hvs.append(channel_hv)

        # 特征融合：将所有通道的超向量进行捆绑（bundle）
        if len(channel_hvs) == 1:
            fused_hv = channel_hvs[0]
        else:
            fused_hv = channel_hvs[0]
            for hv in channel_hvs[1:]:
                fused_hv = torchhd.bundle(fused_hv, hv)

        # 使用n-gram捕获更复杂的时序关系（可选）
        if self.n_gram_size > 1:
            # 将融合后的向量扩展为序列形式用于n-gram
            fused_hv_seq = fused_hv.unsqueeze(1).expand(-1, self.n_gram_size, -1)
            fused_hv = torchhd.ngrams(fused_hv_seq, n=self.n_gram_size)

        # 硬量化，将超维向量转换为二进制向量
        return torchhd.hard_quantize(fused_hv)


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
    
    # 检查标签是否为多通道
    is_multichannel = isinstance(train_ds.labels[0], np.ndarray)
    print(f"检测到 {'多通道' if is_multichannel else '单通道'} 标签")
    
    # 获取所有唯一标签，确保模型包含所有可能的类别
    all_labels = []
    for label in train_ds.labels:
        if is_multichannel:
            all_labels.extend(label)
        else:
            all_labels.append(label)
    
    max_label = max(all_labels)
    num_classes = int(max_label + 1)  # 转换为Python原生int类型
    print(f"检测到 {num_classes} 个唯一类别（标签 0 到 {max_label}）")
    
    # 初始化编码器
    # 样本形状: [downsample_factor, channels]
    sample_shape = train_ds[0][0].shape
    encode = Encoder(DIMENSIONS, sample_shape[0], sample_shape[1], n_gram_size=n_gram_size)
    encode = encode.to(device)
    
    # 初始化模型，使用多个Centroid（质心）模型，每个通道一个
    models = []
    if is_multichannel:
        for channel in range(CHANNELS):
            model = Centroid(DIMENSIONS, num_classes)
            model = model.to(device)
            models.append(model)
    else:
        # 单通道标签，只需要一个模型
        model = Centroid(DIMENSIONS, num_classes)
        model = model.to(device)
        models.append(model)
    
    # 训练模型
    print("正在训练模型...")
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="训练中"):
            samples = samples.to(device)
            targets = targets.to(device)
            
            # 生成超维向量
            sample_hv = encode(samples)
            
            if is_multichannel:
                # 多通道标签，为每个通道更新对应的模型
                for channel in range(CHANNELS):
                    channel_targets = targets[:, channel].long()
                    models[channel].add(sample_hv, channel_targets)
            else:
                # 单通道标签，只需要更新一个模型
                models[0].add(sample_hv, targets.long())
    
    # 归一化模型
    for model in models:
        model.normalize()
    
    # 保存模型和编码器
    state_dict = {
        'encoder_state_dict': encode.state_dict(),
        'model_state_dicts': [model.state_dict() for model in models],
        'num_classes': num_classes,
        'sample_shape': sample_shape,
        'n_gram_size': n_gram_size,
        'is_multichannel': is_multichannel
    }
    torch.save(state_dict, MODEL_PATH)
    
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
    is_multichannel = checkpoint.get('is_multichannel', False)
    print(f"预测时使用模型训练参数: N_GRAM_SIZE={n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP}")
    print(f"模型支持 {'多通道' if is_multichannel else '单通道'} 标签")
    
    # 初始化编码器
    sample_shape = checkpoint['sample_shape']
    encode = Encoder(DIMENSIONS, sample_shape[0], sample_shape[1], n_gram_size=n_gram_size)
    encode.load_state_dict(checkpoint['encoder_state_dict'])
    encode = encode.to(device)
    encode.eval()
    
    num_classes = checkpoint['num_classes']
    
    # 初始化多个模型，每个通道一个
    models = []
    model_state_dicts = checkpoint['model_state_dicts']
    for state_dict in model_state_dicts:
        model = Centroid(DIMENSIONS, num_classes)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    
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
            
            if is_multichannel:
                # 多通道预测，为每个通道生成预测结果
                pred_labels = []
                for channel in range(CHANNELS):
                    # 使用对应通道的模型进行预测
                    output = models[channel](sample_hv, dot=True)
                    # 使用余弦相似度（cosine similarity）进行分类
                    # dot=True返回的是点积，需要转换为余弦相似度
                    similarities = output / (torch.norm(sample_hv, dim=1, keepdim=True) * torch.norm(models[channel].centroids, dim=1))
                    # 获取预测标签（相似度最高的类别）
                    pred_label = torch.argmax(similarities, dim=1).cpu().item()
                    pred_labels.append(pred_label)
                predictions.append(pred_labels)
            else:
                # 单通道预测，使用余弦相似度
                output = models[0](sample_hv, dot=True)
                # 计算余弦相似度
                similarities = output / (torch.norm(sample_hv, dim=1, keepdim=True) * torch.norm(models[0].centroids, dim=1))
                pred_label = torch.argmax(similarities, dim=1).cpu().item()
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
        try:
            # 读取原始文件
            original_file = data_dir / file_name
            df = pd.read_csv(original_file)
            
            # 添加预测结果列
            if is_multichannel:
                # 多通道预测结果，添加predicted_label_0到predicted_label_8列
                for channel in range(CHANNELS):
                    df[f'predicted_label_{channel}'] = df[f'label_{channel}'] if f'label_{channel}' in df.columns else 0
            else:
                # 单通道预测结果，添加predicted_label列
                df['predicted_label'] = df['label'] if 'label' in df.columns else 0
            
            # 根据预测结果更新标签
            for start, end, pred_label in pred_info:
                if is_multichannel:
                    # 多通道预测结果
                    for channel in range(CHANNELS):
                        df.loc[start:end-1, f'predicted_label_{channel}'] = pred_label[channel]
                else:
                    # 单通道预测结果
                    df.loc[start:end-1, 'predicted_label'] = pred_label
            
            # 保存带预测结果的CSV文件，确保只添加一个hdc后缀
            output_file = data_dir / f"{original_file.stem.replace('_hdc', '')}_hdc.csv"
            df.to_csv(output_file, index=False)
            print(f"已保存预测结果到: {output_file}")
        except PermissionError as e:
            print(f"警告: 无法写入文件 {file_name}，权限不足: {e}")
        except Exception as e:
            print(f"警告: 处理文件 {file_name} 时出错: {e}")
    
    print("所有预测结果已保存")
    return True


# ======================================
# 模型评估函数
# ======================================
def evaluate_model(test_dir=TEST_DIR, train_dir=TRAIN_DIR):
    """评估模型性能，使用工业探伤专用的评估指标

    Args:
        test_dir: 测试数据目录
        train_dir: 训练数据目录
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import pandas as pd

    print("开始模型评估...")

    # 收集所有预测结果和真实标签
    all_true_labels = []
    all_pred_labels = []

    # 评估测试数据
    test_files = list(Path(test_dir).glob("*_hdc.csv"))
    for csv_file in test_files:
        try:
            df = pd.read_csv(csv_file)

            # 检查是否有预测结果
            if 'predicted_label' not in df.columns and not any(f'predicted_label_{i}' in df.columns for i in range(9)):
                continue

            # 提取真实标签和预测标签
            if 'label' in df.columns:
                # 单通道标签
                true_labels = df['label'].values
                pred_labels = df['predicted_label'].values if 'predicted_label' in df.columns else df['predicted_label_0'].values
            else:
                # 多通道标签，使用第一个通道作为代表
                true_labels = df['label_0'].values
                pred_labels = df['predicted_label_0'].values

            # 只保留有真实标签的行（非零标签）
            valid_indices = true_labels != 0
            all_true_labels.extend(true_labels[valid_indices])
            all_pred_labels.extend(pred_labels[valid_indices])

        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")
            continue

    if not all_true_labels:
        print("没有找到有效的评估数据")
        return

    # 计算混淆矩阵
    cm = confusion_matrix(all_true_labels, all_pred_labels)

    # 计算各项指标
    def calculate_metrics(cm):
        """计算探伤专用的评估指标"""
        if cm.size == 0:
            return {}

        # 基本指标
        tn = cm[0, 0] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

        # 避免除零错误
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # F2分数（优先保证召回率）
        beta = 2
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

        # 马修斯相关系数 (MCC)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0

        # 漏检率 (False Negative Rate)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # 误报率 (False Positive Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'mcc': mcc,
            'fnr': fnr,
            'fpr': fpr
        }

    metrics = calculate_metrics(cm)

    # 打印评估结果
    print("\n" + "="*60)
    print("=== 模型评估结果（工业探伤专用指标）===")
    print("="*60)
    print("混淆矩阵:")
    print(cm)
    print("\n核心指标:")
    print(".4f" )
    # 评估等级
    if metrics['f2'] >= 0.85 and metrics['mcc'] >= 0.85 and metrics['fnr'] <= 0.01:
        grade = "优秀 (★★★★★)"
        desc = "满足工业探伤要求，可投入使用"
    elif metrics['f2'] >= 0.80 and metrics['mcc'] >= 0.70:
        grade = "良好 (★★★★☆)"
        desc = "性能良好，建议进一步优化"
    elif metrics['f2'] >= 0.70:
        grade = "一般 (★★★☆☆)"
        desc = "基本可用，需要算法优化"
    else:
        grade = "需优化 (★★☆☆☆)"
        desc = "性能不足，建议重新调整参数"

    print(f"\n模型等级: {grade}")
    print(f"评估结论: {desc}")

    return metrics


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

    # 评估模型
    evaluate_model()

    print("\n=== 训练、预测和评估完成 ===")


if __name__ == "__main__":
    # 运行主函数
    main()