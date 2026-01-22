#!/usr/bin/env python3
"""只使用通道8进行训练和识别的实验"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# 添加torchhd库的父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'torchhd'))

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

# 导入超声波数据转换器
from ultrasonic_data_converter import UltrasonicDataConverter

# ======================================
# 设备设置
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ======================================
# 配置参数
# ======================================
# 超向量维度
DIMENSIONS = 10000
# 水平嵌入的级别数
NUM_LEVELS = 256
# 批处理大小
BATCH_SIZE = 1

# 默认参数（可以从命令行覆盖）
DEFAULT_WINDOW = 256
DEFAULT_STEP = 1
DEFAULT_N_GRAM_SIZE = 15

# 窗口大小
WINDOW = DEFAULT_WINDOW
# 滑动窗口步长（从1改为8，减少重叠预测）
STEP = DEFAULT_STEP
# N-gram大小
N_GRAM_SIZE = DEFAULT_N_GRAM_SIZE

# 数据目录
TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
TEST_DATA_DIR = Path(r"d:\HDC\data\test")
# 模型保存路径
MODEL_PATH = Path(r"d:\HDC\models\hdc_model_channel8.pth")
MODEL_PATH.parent.mkdir(exist_ok=True)

# ======================================
# 通道8数据集类
# ======================================TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
TEST_DATA_DIR = Path(r"d:\HDC\data\test")
# 模型保存路径
MODEL_PATH = Path(r"d:\HDC\models\hdc_model_channel8.pth")
MODEL_PATH.parent.mkdir(exist_ok=True)


# ======================================
# HDC数据集类
# ======================================
class HDCDataset(data.Dataset):
    """多通道HDC数据集类
    
    Args:
        data_dir: 数据目录路径
        is_train: 是否为训练模式
    """
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        self.samples = []
        self.labels = []
        self.file_info = []
        
        self._load_all_data()
    
    def _load_all_data(self):
        """加载所有数据并生成样本"""
        print(f"正在加载 {self.data_dir} 目录下的所有CSV文件")
        
        # 查找 _hdc.csv 文件
        csv_files = list(self.data_dir.glob("*_hdc.csv"))
        print(f"找到 {len(csv_files)} 个 _hdc.csv 文件")
        
        for csv_file in csv_files:
            print(f"\n正在处理文件: {csv_file.name}")

            df = pd.read_csv(csv_file)
            
            # 查找对应的 label 文件
            base_name = csv_file.name.replace("_hdc.csv", "")
            label_file = self.data_dir / f"{base_name}_label.csv"
            
            if label_file.exists():
                label_df = pd.read_csv(label_file)
                # 合并数据和标签
                df = pd.merge(df, label_df, on='encoding', how='left').fillna(0)
                print(f"已合并标签文件: {label_file.name}")
            else:
                print(f"警告: 未找到标签文件 {label_file.name}，使用默认标签0")
                for i in range(9):
                    df[f'label_{i}'] = 0

            # 提取 0-8 通道的声程数据
            channel_cols = [f'channel_{i}' for i in range(9)]
            label_cols = [f'label_{i}' for i in range(9)]
            
            missing_cols = [col for col in channel_cols if col not in df.columns]
            if missing_cols:
                print(f"警告: {csv_file.name} 缺少必要的列 {missing_cols}，跳过")
                continue

            df = df.sort_values(by='encoding')
            
            # 提取所有通道的数据 (N, 9)
            all_channel_data = df[channel_cols].values.astype(np.float32)
            # 提取所有通道的标签 (N, 9)
            all_label_data = df[label_cols].values.astype(np.int64)
            
            encodings = df['encoding'].values
            
            if len(encodings) < WINDOW:
                print(f"警告: 数据不足，无法生成窗口。需要 {WINDOW} 个编码值，仅获得 {len(encodings)} 个")
                continue
            
            # 生成滑动窗口样本
            file_samples = []
            file_labels = []
            file_indices = []
            
            for start in range(0, len(encodings) - WINDOW + 1, STEP):
                end = start + WINDOW
                
                # 窗口数据 (WINDOW, 9)
                window_data = all_channel_data[start:end]
                sample = window_data.astype(np.float32)
                
                center = (start + end) // 2
                
                # 确定窗口标签
                # 修改：考虑中心点附近一个小范围（例如前后2个点）的标签
                # 如果该范围内有任何非零标签，取出现频率最高的那个
                search_radius = 2
                range_start = max(0, center - search_radius)
                range_end = min(len(all_label_data), center + search_radius + 1)
                
                neighborhood_labels = all_label_data[range_start:range_end].flatten()
                neighborhood_labels = neighborhood_labels[neighborhood_labels != 0]
                
                if len(neighborhood_labels) > 0:
                    # 取众数作为标签
                    window_label = np.bincount(neighborhood_labels).argmax()
                else:
                    window_label = 0
                
                file_samples.append(sample)
                file_labels.append(window_label)
                file_indices.append((str(csv_file), start, end, encodings[center]))
            
            self.samples.extend(file_samples)
            self.labels.extend(file_labels)
            self.file_info.extend(file_indices)
            
            print(f"从 {csv_file.name} 生成了 {len(file_samples)} 个窗口样本")
        
        print(f"\n所有文件处理完成，共生成 {len(self.samples)} 个窗口样本")
        if self.labels:
            label_counts = np.bincount(self.labels)
            print(f"标签分布: {label_counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        sample = torch.from_numpy(sample)
        return sample, label


# ======================================
# HDC编码器类（多通道版本）
# ======================================
class HDCEncoder(nn.Module):
    """多通道超维编码器
    
    Args:
        out_features: 输出超维向量的维度
        timestamps: 输入时序数据的时间戳数量
        n_gram_size: n-gram大小
    """
    def __init__(self, out_features, timestamps, n_gram_size=N_GRAM_SIZE):
        super(HDCEncoder, self).__init__()
        
        # 通道嵌入层（为 0-8 通道创建嵌入）
        self.channels = embeddings.Random(9, out_features)
        # 信号嵌入层（调整范围以适应实际数据）
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=256)
        # n-gram大小
        self.n_gram_size = n_gram_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            input: 输入数据，形状为 [batch, timestamps, 9]
            
        Returns:
            超维向量，形状为 [batch, out_features]
        """
        # input: [batch, timestamps, 9]
        # signals(input): [batch, timestamps, 9, out_features]
        signal_hvs = self.signals(input)
        
        # channels.weight: [9, out_features]
        # 需要 unsqueeze 以匹配 signal_hvs 的维度进行绑定
        # channel_hv: [1, 1, 9, out_features]
        channel_hv = self.channels.weight.unsqueeze(0).unsqueeze(0)
        
        # 绑定信号和通道信息
        # bound_hvs: [batch, timestamps, 9, out_features]
        bound_hvs = torchhd.bind(signal_hvs, channel_hv)
        
        # 在通道维度上聚合: [batch, timestamps, out_features]
        # 使用 multiset 聚合一个时间戳内所有通道的信息
        aggregated_hvs = torchhd.multiset(bound_hvs)
        
        # 使用 n-gram 捕获时间序列关系: [batch, out_features]
        sample_hv = torchhd.ngrams(aggregated_hvs, n=self.n_gram_size)
        
        # 归一化/量化
        return torchhd.normalize(sample_hv)


# ======================================
# 训练函数
# ======================================
def train_hdc_model(n_gram_size=N_GRAM_SIZE):
    """使用多通道数据训练HDC模型
    
    Args:
        n_gram_size: N-gram大小
    """
    print(f"开始训练HDC模型... (N_GRAM_SIZE: {n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP})")
    
    train_ds = HDCDataset(TRAIN_DATA_DIR, is_train=True)
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    if len(train_ds) == 0:
        print("错误: 训练数据集为空！")
        return False
    
    # 固定使用20个类别（0-19：正常+19种伤损类型）
    num_classes = 20
    print(f"使用固定类别数: {num_classes}（标签 0 到 19）")
    
    # 初始化编码器
    sample_shape = train_ds[0][0].shape
    encode = HDCEncoder(DIMENSIONS, sample_shape[0], n_gram_size=n_gram_size)
    encode = encode.to(device)
    
    # 初始化模型
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    
    # 训练模型
    print("正在训练模型...")
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="训练中"):
            samples = samples.to(device)
            targets = targets.to(device)
            
            sample_hv = encode(samples)
            model.add(sample_hv, targets.long())
    
    # 归一化模型
    model.normalize()
    
    # 计算训练准确率
    print("正在计算训练准确率...")
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for samples, targets in tqdm(train_ld, desc="计算训练准确率"):
            samples = samples.to(device)
            targets = targets.to(device)
            
            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            pred_label = torch.argmax(output, dim=1)
            
            train_correct += (pred_label == targets).sum().item()
            train_total += targets.size(0)
    
    train_accuracy = (train_correct / train_total) * 100 if train_total > 0 else 0
    print(f"训练准确率: {train_accuracy:.2f}%")
    
    # 保存模型和编码器
    state_dict = {
        'encoder_state_dict': encode.state_dict(),
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'sample_shape': sample_shape,
        'n_gram_size': n_gram_size
    }
    torch.save(state_dict, MODEL_PATH)
    
    print(f"模型训练完成，已保存到: {MODEL_PATH}")
    return True


# ======================================
# 预测函数
# ======================================
def predict_hdc_model(data_dir=TEST_DATA_DIR):
    """使用训练好的模型对数据进行预测
    
    Args:
        data_dir: 预测数据目录
    """
    print("开始模型预测...")
    
    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return False
    
    # 加载模型和编码器
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    
    n_gram_size = checkpoint.get('n_gram_size', N_GRAM_SIZE)
    print(f"预测时使用模型训练参数: N_GRAM_SIZE={n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP}")
    
    # 初始化编码器
    sample_shape = checkpoint['sample_shape']
    encode = HDCEncoder(DIMENSIONS, sample_shape[0], n_gram_size=n_gram_size)
    encode.load_state_dict(checkpoint['encoder_state_dict'])
    encode = encode.to(device)
    encode.eval()
    
    num_classes = checkpoint['num_classes']
    
    # 初始化模型
    model = Centroid(DIMENSIONS, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载数据集
    test_ds = HDCDataset(data_dir, is_train=False)
    test_ld = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(test_ds) == 0:
        print(f"错误: {data_dir} 目录下的数据集为空！")
        return False
    
    # 执行预测
    print(f"正在对 {data_dir} 目录下的数据进行预测...")
    predictions = []
    with torch.no_grad():
        for samples, targets in tqdm(test_ld, desc="预测中"):
            samples = samples.to(device)
            
            sample_hv = encode(samples)
            output = model(sample_hv, dot=True)
            pred_label = torch.argmax(output, dim=1).cpu().item()
            predictions.append(pred_label)
    
    # 保存预测结果为 _predicted_label.csv
    print("正在保存预测结果...")
    
    file_predictions = {}
    for idx, pred_label in enumerate(predictions):
        file_path, start, end, center_encoding = test_ds.file_info[idx]
        file_key = Path(file_path).name
        if file_key not in file_predictions:
            file_predictions[file_key] = []
        file_predictions[file_key].append((center_encoding, pred_label))
    
    for file_name, pred_info in file_predictions.items():
        try:
            # 读取原始 _hdc.csv 以获取完整的 encoding 列表
            original_hdc_file = data_dir / file_name
            df_hdc = pd.read_csv(original_hdc_file)
            
            # 创建预测标签 DataFrame
            # 初始化所有通道的预测标签为0
            pred_df = pd.DataFrame({'encoding': df_hdc['encoding']})
            for i in range(9):
                pred_df[f'predicted_label_{i}'] = 0
            
            # 填充预测结果
            # 改进：只将预测标签应用到有信号的通道
            for center_encoding, pred_label in pred_info:
                # 找到对应的行
                mask = pred_df['encoding'] == center_encoding
                if mask.any() and pred_label != 0:
                    # 获取该行原始数据
                    row_idx = pred_df.index[mask][0]
                    # 检查哪些通道有信号（声程 > 0）
                    for i in range(9):
                        if df_hdc.loc[row_idx, f'channel_{i}'] > 0:
                            pred_df.loc[mask, f'predicted_label_{i}'] = pred_label
            
            # 保存为 _predicted_label.csv
            base_name = file_name.replace("_hdc.csv", "")
            output_file = data_dir / f"{base_name}_predicted_label.csv"
            try:
                pred_df.to_csv(output_file, index=False)
                print(f"已保存预测结果到: {output_file}")
            except PermissionError:
                print(f"警告: 无法保存预测结果到 {output_file}，文件可能已被其他程序占用。")
            
            # 同时更新原始 _hdc.csv 以便后续计算准确率（保持兼容性）
            try:
                for i in range(9):
                    df_hdc[f'predicted_label_{i}'] = pred_df[f'predicted_label_{i}']
                df_hdc.to_csv(original_hdc_file, index=False)
            except PermissionError:
                print(f"警告: 无法更新原始文件 {original_hdc_file}，文件可能已被其他程序占用。")
            
        except Exception as e:
            print(f"警告: 处理文件 {file_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("所有预测结果已保存")
    return True


# ======================================
# 准确率计算函数
# ======================================
def calculate_hdc_accuracy(predicted_file, original_file):
    """计算HDC识别正确率（多通道）
    
    Args:
        predicted_file: 包含预测结果的CSV文件路径（或包含预测列的hdc.csv）
        original_file: 原始CSV文件路径（包含真实标签，通常是 _hdc.csv）
    
    Returns:
        dict: 包含精确率、召回率和F1分数的字典
    """
    try:
        pred_df = pd.read_csv(predicted_file)
        orig_df = pd.read_csv(original_file)
        
        # 查找对应的 label 文件以获取真实标签
        base_name = Path(original_file).name.replace("_hdc.csv", "")
        label_file = Path(original_file).parent / f"{base_name}_label.csv"
        
        if label_file.exists():
            label_df = pd.read_csv(label_file)
            orig_df = pd.merge(orig_df, label_df, on='encoding', how='left').fillna(0)
        
        # 检查必要的列
        pred_cols = [f'predicted_label_{i}' for i in range(9)]
        true_cols = [f'label_{i}' for i in range(9)]
        
        available_pred_cols = [col for col in pred_cols if col in pred_df.columns]
        available_true_cols = [col for col in true_cols if col in orig_df.columns]
        
        if not available_pred_cols or not available_true_cols:
            print(f"警告: 缺少预测标签或真实标签列，无法计算正确率")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 合并数据
        merged_df = pd.merge(orig_df[['encoding'] + available_true_cols], 
                             pred_df[['encoding'] + available_pred_cols], 
                             on='encoding')
        
        if merged_df.empty:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # 展平所有通道的数据进行整体评估
        y_true = merged_df[available_true_cols].values.flatten()
        y_pred = merged_df[available_pred_cols].values.flatten()
        
        # 计算统计量
        total_identified = np.sum(y_pred != 0)
        total_true_defects = np.sum(y_true != 0)
        total_correct = np.sum((y_pred != 0) & (y_true != 0))
        
        precision = total_correct / total_identified if total_identified > 0 else 0.0
        recall = total_correct / total_true_defects if total_true_defects > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_identified': total_identified,
            'total_true_defects': total_true_defects,
            'total_correct': total_correct
        }
    except Exception as e:
        print(f"计算准确率时出错: {e}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'total_identified': 0, 'total_true_defects': 0, 'total_correct': 0}


# ======================================
# 清理旧文件
# ======================================
def clean_old_files(data_dirs):
    """清理旧的临时文件"""
    for data_dir in data_dirs:
        print(f"清理目录: {data_dir}")
        patterns = ["*_hdc.csv", "*_label.csv", "*_encoding.csv", "*.csv", "*_predicted_label.csv", "*_predicted.json"]
        for pattern in patterns:
            for file_path in data_dir.glob(pattern):
                # 排除原始 .json 文件
                if file_path.suffix == '.json' and not file_path.name.endswith('_predicted.json') and '标注' not in file_path.name:
                    continue
                try:
                    file_path.unlink()
                    print(f"✓ 删除文件: {file_path.name}")
                except Exception as e:
                    pass


# ======================================
# 数据转换
# ======================================
def get_data_files(data_dir):
    """获取数据目录下的所有数据文件"""
    data_files = []
    for ext in ['*.json', '*.xxc.json']:
        data_files.extend(data_dir.glob(ext))
    return [f for f in data_files if '标注' not in f.name]


def convert_data_files(data_files, converter):
    """转换数据文件为CSV格式"""
    csv_files = []
    for data_file in data_files:
        annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
        annotation_path = str(annotation_file) if annotation_file.exists() else None
        output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
        
        try:
            result_file = converter.convert_single_file(str(data_file), annotation_path, str(output_csv))
            csv_files.append(str(output_csv))
            print(f"✓ 转换完成: {data_file.name} {'(使用标注文件)' if annotation_path else '(无标注文件)'}")
        except Exception as e:
            print(f"✗ 转换失败 {data_file.name}: {e}")
            continue
    return csv_files


# ======================================
# 主测试函数
# ======================================
def test_full_hdc_pipeline():
    """测试多通道HDC完整流程"""
    print("=== 开始多通道HDC实验 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    OUTPUT_DIR = Path("hdc_test_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 清理旧文件
    print("\n=== 清理旧文件 ===")
    clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR])
    print("=== 清理完成 ===")
    
    try:
        # 1. 数据转换 (JSON -> 1.csv, 1_encoding.csv, 1_hdc.csv, 1_label.csv)
        print("\n=== Step 1: 数据转换 ===")
        converter = UltrasonicDataConverter()
        
        for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
            data_files = get_data_files(data_dir)
            for data_file in data_files:
                try:
                    annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
                    annotation_path = str(annotation_file) if annotation_file.exists() else None
                    
                    print(f"正在转换文件: {data_file.name}")
                    converter.generate_all_csvs(str(data_file), annotation_path)
                    print(f"✓ 转换完成: {data_file.name}")
                except Exception as e:
                    print(f"✗ 转换失败 {data_file.name}: {e}")
        
        # 2. 模型训练
        print("\n=== Step 2: 模型训练 ===")
        if not train_hdc_model():
            print("[FAIL] 模型训练失败")
            return False
        
        # 3. 模型预测 (HDC CSV -> _predicted_label.csv)
        print("\n=== Step 3: 模型预测 ===")
        for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
            if not predict_hdc_model(data_dir):
                print(f"[FAIL] {data_dir} 模型预测失败")
                return False
        
        # 4. CSV -> JSON 转换 (_predicted_label.csv -> _predicted.json)
        print("\n=== Step 4: CSV到HDC JSON转换 ===")
        try:
            from csv_to_hdc_json import CsvToHdcJsonConverter
            json_converter = CsvToHdcJsonConverter()
            
            for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
                # 查找所有 _predicted_label.csv 文件
                pred_csvs = list(data_dir.glob("*_predicted_label.csv"))
                for pred_csv in pred_csvs:
                    # 查找对应的 _hdc.csv 以获取深度信息
                    base_name = pred_csv.name.replace("_predicted_label.csv", "")
                    hdc_csv = data_dir / f"{base_name}_hdc.csv"
                    
                    if not hdc_csv.exists():
                        print(f"警告: 找不到对应的 HDC CSV 文件 {hdc_csv.name}，跳过 JSON 转换")
                        continue
                        
                    # 转换
                    df_pred = pd.read_csv(pred_csv)
                    df_hdc = pd.read_csv(hdc_csv)
                    # 合并以获取完整信息
                    merged_df = pd.merge(df_hdc, df_pred, on='encoding')
                    
                    # 聚类并生成 JSON
                    clustered_df = json_converter.cluster_points(merged_df)
                    hdc_json = json_converter.generate_hdc_json(clustered_df, str(hdc_csv))
                    
                    # 保存 JSON
                    output_json = data_dir / f"{base_name}_predicted.json"
                    import json
                    with open(output_json, 'w', encoding='utf-8') as f:
                        json.dump(hdc_json, f, ensure_ascii=False, indent=2)
                    print(f"✓ 已生成预测 JSON: {output_json.name}")
        except Exception as e:
            print(f"[FAIL] CSV到HDC JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. 计算准确率
        print("\n=== Step 5: 计算准确率 ===")
        all_metrics = []
        for data_dir in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
            data_files = get_data_files(data_dir)
            for data_file in data_files:
                original_hdc = data_dir / f"{data_file.stem}_hdc.csv"
                predicted_label = data_dir / f"{data_file.stem}_predicted_label.csv"
                
                if original_hdc.exists() and predicted_label.exists():
                    metrics = calculate_hdc_accuracy(predicted_label, original_hdc)
                    metrics['file'] = data_file.name
                    metrics['type'] = 'train' if data_dir == TRAIN_DATA_DIR else 'test'
                    all_metrics.append(metrics)
                    
                    print(f"  {data_file.name} ({metrics['type']}):")
                    print(f"    精确率 = {metrics['precision']:.4f}, 召回率 = {metrics['recall']:.4f}, F1 = {metrics['f1']:.4f}")
        
        if all_metrics:
            avg_precision = np.mean([m['precision'] for m in all_metrics])
            avg_recall = np.mean([m['recall'] for m in all_metrics])
            avg_f1 = np.mean([m['f1'] for m in all_metrics])
            print(f"\n  平均精确率: {avg_precision:.4f}")
            print(f"  平均召回率: {avg_recall:.4f}")
            print(f"  平均F1分数: {avg_f1:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='通道8专用HDC实验')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW, 
                       help=f'窗口大小（默认: {DEFAULT_WINDOW}）')
    parser.add_argument('--step', type=int, default=DEFAULT_STEP, 
                       help=f'滑动步长（默认: {DEFAULT_STEP}）')
    parser.add_argument('--n-gram', type=int, default=DEFAULT_N_GRAM_SIZE, 
                       help=f'N-gram大小（默认: {DEFAULT_N_GRAM_SIZE}）')
    parser.add_argument('--no-convert', action='store_true', 
                       help='跳过数据转换，直接使用已有的CSV文件')
    
    args = parser.parse_args()
    
    # 更新全局参数
    global WINDOW, STEP, N_GRAM_SIZE
    WINDOW = args.window
    STEP = args.step
    N_GRAM_SIZE = args.n_gram
    
    print("=== 通道8专用HDC实验 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用的窗口大小: {WINDOW}")
    print(f"使用的滑动步长: {STEP}")
    print(f"使用的N_GRAM_SIZE: {N_GRAM_SIZE}")
    
    success = test_full_hdc_pipeline()
    
    if success:
        print("\n✓ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
