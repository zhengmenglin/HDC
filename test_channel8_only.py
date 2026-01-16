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
NUM_LEVELS = 255
# 批处理大小
BATCH_SIZE = 1

# 默认参数（可以从命令行覆盖）
DEFAULT_WINDOW = 20
DEFAULT_STEP = 20
DEFAULT_N_GRAM_SIZE = 4

# 窗口大小
WINDOW = DEFAULT_WINDOW
# 滑动窗口步长（从1改为2，减少重叠预测）
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
# ======================================
class Channel8Dataset(data.Dataset):
    """只使用通道8的数据集类
    
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
        
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in csv_files:
            print(f"\n正在处理文件: {csv_file.name}")
            
            df = pd.read_csv(csv_file)
            df = df.sort_values(by='encoding')
            
            # 只提取通道8的数据
            if 'channel_8' not in df.columns:
                print(f"警告: {csv_file.name} 没有channel_8列，跳过")
                continue
            
            channel_data = df['channel_8'].values.reshape(-1, 1).astype(np.float32)
            
            # 提取通道8的标签
            if 'label_8' in df.columns:
                labels = df['label_8'].values
                print(f"使用label_8标签，形状: {labels.shape}")
            else:
                print(f"警告: {csv_file.name} 没有label_8列，使用默认标签0")
                labels = np.zeros(len(df))
            
            encodings = df['encoding'].values
            print(f"从 {csv_file.name} 加载了 {len(encodings)} 个编码值")
            
            if len(encodings) < WINDOW:
                print(f"警告: 数据不足，无法生成窗口。需要 {WINDOW} 个编码值，仅获得 {len(encodings)} 个")
                continue
            
            # 生成滑动窗口样本
            file_samples = []
            file_labels = []
            file_indices = []
            
            for start in range(0, len(encodings) - WINDOW + 1, STEP):
                end = start + WINDOW
                
                window_data = channel_data[start:end]
                sample = window_data.astype(np.float32)
                
                # 确定窗口标签
                window_labels = labels[start:end]
                if len(window_labels) > 0:
                    label = max(set(window_labels), key=list(window_labels).count)
                else:
                    label = 0
                
                file_samples.append(sample)
                file_labels.append(label)
                file_indices.append((str(csv_file), start, end))
            
            self.samples.extend(file_samples)
            self.labels.extend(file_labels)
            self.file_info.extend(file_indices)
            
            print(f"从 {csv_file.name} 生成了 {len(file_samples)} 个窗口样本")
        
        print(f"\n所有文件处理完成，共生成 {len(self.samples)} 个窗口样本")
        if self.labels:
            print(f"标签分布: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        sample = torch.from_numpy(sample)
        return sample, label


# ======================================
# 超维编码器类（单通道版本）
# ======================================
class Channel8Encoder(nn.Module):
    """单通道超维编码器
    
    Args:
        out_features: 输出超维向量的维度
        timestamps: 输入时序数据的时间戳数量
        n_gram_size: n-gram大小
    """
    def __init__(self, out_features, timestamps, n_gram_size=N_GRAM_SIZE):
        super(Channel8Encoder, self).__init__()
        
        # 信号嵌入层
        self.signals = embeddings.Level(NUM_LEVELS, out_features, high=20)
        # n-gram大小
        self.n_gram_size = n_gram_size
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            input: 输入数据，形状为 [batch, timestamps, 1]
            
        Returns:
            超维向量，形状为 [batch, out_features]
        """
        # 将连续信号嵌入到超维空间
        signal = self.signals(input)
        
        # 使用multiset聚合样本
        samples = torchhd.multiset(signal)
        # 使用n-gram捕获时序关系
        sample_hv = torchhd.ngrams(samples, n=self.n_gram_size)
        # 硬量化
        return torchhd.hard_quantize(sample_hv)


# ======================================
# 训练函数
# ======================================
def train_channel8_model(n_gram_size=N_GRAM_SIZE):
    """使用通道8的数据训练HDC模型
    
    Args:
        n_gram_size: N-gram大小
    """
    print(f"开始训练通道8模型... (N_GRAM_SIZE: {n_gram_size}, 窗口大小: {WINDOW}, 滑动步长: {STEP})")
    
    train_ds = Channel8Dataset(TRAIN_DATA_DIR, is_train=True)
    train_ld = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    if len(train_ds) == 0:
        print("错误: 训练数据集为空！")
        return False
    
    # 固定使用20个类别（0-19：正常+19种伤损类型）
    num_classes = 20
    print(f"使用固定类别数: {num_classes}（标签 0 到 19）")
    
    # 初始化编码器
    sample_shape = train_ds[0][0].shape
    encode = Channel8Encoder(DIMENSIONS, sample_shape[0], n_gram_size=n_gram_size)
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
def predict_channel8_model(data_dir=TEST_DATA_DIR):
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
    encode = Channel8Encoder(DIMENSIONS, sample_shape[0], n_gram_size=n_gram_size)
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
    test_ds = Channel8Dataset(data_dir, is_train=False)
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
    
    # 保存预测结果
    print("正在保存预测结果...")
    
    file_predictions = {}
    for idx, pred_label in enumerate(predictions):
        file_path, start, end = test_ds.file_info[idx]
        file_key = Path(file_path).name
        if file_key not in file_predictions:
            file_predictions[file_key] = []
        file_predictions[file_key].append((start, end, pred_label))
    
    for file_name, pred_info in file_predictions.items():
        try:
            original_file = data_dir / file_name
            df = pd.read_csv(original_file)
            
            # 添加predicted_label_8列
            if 'predicted_label_8' not in df.columns:
                df['predicted_label_8'] = df['label_8'] if 'label_8' in df.columns else 0
            
            # 更新预测结果（只标记窗口中心点，减少误报）
            for start, end, pred_label in pred_info:
                center = (start + end) // 2  # 计算窗口中心点
                df.loc[center, 'predicted_label_8'] = pred_label
            
            output_file = data_dir / f"{original_file.stem.replace('_hdc', '')}_hdc.csv"
            df.to_csv(output_file, index=False)
            print(f"已保存预测结果到: {output_file}")
        except Exception as e:
            print(f"警告: 处理文件 {file_name} 时出错: {e}")
    
    print("所有预测结果已保存")
    return True


# ======================================
# 准确率计算函数
# ======================================
def calculate_channel8_accuracy(predicted_file, original_file):
    """计算通道8的识别正确率
    
    计算精确率、召回率和F1分数：
    - 精确率 = 预测正确的点数 / 所有预测出的点数
    - 召回率 = 预测正确的点数 / 所有真实伤损点数
    - F1分数 = 2 * 精确率 * 召回率 / (精确率 + 召回率)
    
    Args:
        predicted_file: 包含预测结果的CSV文件路径
        original_file: 原始CSV文件路径（包含真实标签）
    
    Returns:
        dict: 包含精确率、召回率和F1分数的字典
    """
    try:
        pred_df = pd.read_csv(predicted_file)
        orig_df = pd.read_csv(original_file)
        
        if 'label_8' not in orig_df.columns or 'predicted_label_8' not in pred_df.columns:
            print(f"警告: 缺少label_8或predicted_label_8列，无法计算正确率")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if len(pred_df) != len(orig_df):
            print(f"警告: 预测结果和原始文件行数不一致，无法计算正确率")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        merged_df = pd.merge(orig_df, pred_df, on='encoding', suffixes=('_orig', '_pred'))
        
        if merged_df.empty:
            print(f"警告: 合并后的数据为空，无法计算正确率")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_labels = merged_df['label_8_orig'].values
        pred_labels = merged_df['predicted_label_8_pred'].values
        
        # 计算各种统计量
        total_identified = np.sum(pred_labels != 0)  # 预测为伤损的点数
        total_true_defects = np.sum(true_labels != 0)  # 真实伤损点数
        total_correct = np.sum((pred_labels != 0) & (true_labels != 0))  # 预测正确的点数
        
        # 计算精确率
        if total_identified == 0:
            precision = 0.0
            print(f"注意: 没有识别出任何点数")
        else:
            precision = total_correct / total_identified
        
        # 计算召回率
        if total_true_defects == 0:
            recall = 0.0
            print(f"注意: 没有真实伤损点")
        else:
            recall = total_correct / total_true_defects
        
        # 计算F1分数
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        # 计算F2分数（召回率权重是精确率的2倍）
        if (4 * precision + recall) == 0:
            f2 = 0.0
        else:
            f2 = 5 * precision * recall / (4 * precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'total_identified': total_identified,
            'total_true_defects': total_true_defects,
            'total_correct': total_correct
        }
    except Exception as e:
        print(f"计算准确率时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'precision': 0.0, 
            'recall': 0.0, 
            'f1': 0.0,
            'total_identified': 0,
            'total_true_defects': 0,
            'total_correct': 0
        }


# ======================================
# 清理旧文件
# ======================================
def clean_old_files(data_dirs):
    """清理旧的hdc.json和hdc.csv文件"""
    for data_dir in data_dirs:
        print(f"清理目录: {data_dir}")
        for file_path in data_dir.glob("*_hdc.csv"):
            try:
                file_path.unlink()
                print(f"✓ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path.name}: {e}")
        for file_path in data_dir.glob("*_hdc.json"):
            try:
                file_path.unlink()
                print(f"✓ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path.name}: {e}")


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
def test_channel8_only():
    """测试只使用通道8的完整流程"""
    print("=== 开始通道8专用实验 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    OUTPUT_DIR = Path("channel8_test_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 清理旧文件
    print("\n=== 清理旧文件 ===")
    clean_old_files([TRAIN_DATA_DIR])
    print("=== 清理完成 ===")
    
    try:
        # 1. 数据转换
        print("\n=== Step 1: 数据转换 ===")
        converter = UltrasonicDataConverter()
        
        train_files = get_data_files(TRAIN_DATA_DIR)
        train_csv_files = convert_data_files(train_files, converter)
        
        if not train_csv_files:
            print("[FAIL] 没有成功转换的训练数据文件")
            return False
        
        # 2. 模型训练
        print("\n=== Step 2: 模型训练 ===")
        try:
            if not train_channel8_model():
                print("[FAIL] 模型训练失败")
                return False
            print("✓ 模型训练成功")
        except Exception as e:
            print(f"[FAIL] 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. 模型预测
        print("\n=== Step 3: 模型预测 ===")
        print("预测训练数据...")
        try:
            if not predict_channel8_model(TRAIN_DATA_DIR):
                print("[FAIL] 模型预测失败")
                return False
            print("✓ 训练数据预测完成")
        except Exception as e:
            print(f"[FAIL] 模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3.5 CSV到HDC JSON转换
        print("\n=== Step 3.5: CSV到HDC JSON转换 ===")
        try:
            from csv_to_hdc_json import CsvToHdcJsonConverter
            converter = CsvToHdcJsonConverter()
            converter.batch_convert(str(TRAIN_DATA_DIR), target_channel=8)
            print("✓ CSV到HDC JSON转换完成（预测点已添加source标识）")
        except Exception as e:
            print(f"[FAIL] CSV到HDC JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 计算准确率
        print("\n=== Step 4: 计算准确率 ===")
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_f2 = 0.0
        file_count = 0
        
        for data_file in train_files:
            if '标注' in data_file.name:
                continue
            
            original_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            predicted_csv = data_file.parent / f"{data_file.stem.replace('_hdc', '')}_hdc.csv"
            
            if not original_csv.exists() or not predicted_csv.exists():
                print(f"✗ 缺少文件，无法计算 {data_file.name} 的准确率")
                continue
            
            metrics = calculate_channel8_accuracy(predicted_csv, original_csv)
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            total_f2 += metrics['f2']
            file_count += 1
            
            print(f"  {data_file.name}:")
            print(f"    精确率 = {metrics['precision']:.4f}")
            print(f"    召回率 = {metrics['recall']:.4f}")
            print(f"    F1分数 = {metrics['f1']:.4f}")
            print(f"    F2分数 = {metrics['f2']:.4f}")
            print(f"    预测点数: {metrics['total_identified']}, 真实伤损点数: {metrics['total_true_defects']}, 正确点数: {metrics['total_correct']}")
        
        avg_precision = total_precision / file_count if file_count > 0 else 0.0
        avg_recall = total_recall / file_count if file_count > 0 else 0.0
        avg_f1 = total_f1 / file_count if file_count > 0 else 0.0
        avg_f2 = total_f2 / file_count if file_count > 0 else 0.0
        
        print(f"\n  平均精确率: {avg_precision:.4f}")
        print(f"  平均召回率: {avg_recall:.4f}")
        print(f"  平均F1分数: {avg_f1:.4f}")
        print(f"  平均F2分数: {avg_f2:.4f}")
        
        # 5. 输出结果
        print("\n" + "="*50)
        print("=== 通道8专用实验结果 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"使用的窗口大小: {WINDOW}")
        print(f"使用的滑动步长: {STEP}")
        print(f"使用的N_GRAM_SIZE: {N_GRAM_SIZE}")
        print(f"通道8平均精确率: {avg_precision:.4f}")
        print(f"通道8平均召回率: {avg_recall:.4f}")
        print(f"通道8平均F1分数: {avg_f1:.4f}")
        print(f"通道8平均F2分数: {avg_f2:.4f}")
        print("="*50)
        
        # 保存结果
        results_df = pd.DataFrame([{
            'window': WINDOW,
            'step': STEP,
            'n_gram': N_GRAM_SIZE,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'f2': avg_f2
        }])
        results_file = OUTPUT_DIR / "channel8_accuracy_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n测试结果已保存到: {results_file}")
        
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
    
    success = test_channel8_only()
    
    if success:
        print("\n✓ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
