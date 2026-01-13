"""
超声波数据转换脚本
将JSON格式的超声波数据转换为CSV格式（类似EMG数据格式）
用于HDC算法训练和识别
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
import argparse


class UltrasonicDataConverter:
    """
    超声波数据转换器
    将JSON格式数据转换为HDC算法所需的CSV格式
    """

    def __init__(self):
        # 伤损类型映射 (根据需求文档)
        self.flaw_types = {
            "轨头外侧核伤": 1,
            "轨头中心核伤": 2,
            "轨头内侧核伤": 3,
            "轨头内、外、中心核伤": 4,
            "轨头内、中心核伤": 5,
            "轨头外、中心核伤": 6,
            "轨头水平分裂": 7,
            "轨头、轨腰分裂": 8,
            "轨腰水平裂": 9,
            "轨底断裂": 10,
            "螺孔裂纹": 11
        }

        # 正常结构类型映射
        self.normal_types = {
            "接头": 101,
            "螺孔": 102,
            "导线孔": 103,
            "焊缝": 104,
            "鱼鳞伤": 105,
            "轨底锈蚀": 106,
            "轨底焊筋": 107,
            "道岔": 108
        }

        # 通道映射 (0-8对应实际的通道1-9)
        self.channel_mapping = {i: i+1 for i in range(9)}

    def load_json_data(self, json_file: str) -> Tuple[List[Dict], Dict]:
        """
        加载JSON格式的超声波原始数据

        Args:
            json_file: JSON文件路径

        Returns:
            data_list: 数据列表
            config: 配置信息
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_list = []
        config = {}

        # 提取配置信息（第一个元素通常是配置）
        if data and isinstance(data[0], dict):
            config_item = data[0]
            if 'E' in config_item and isinstance(config_item['E'], int) and config_item['E'] == 1:
                config = config_item
                data_list = data[1:]  # 跳过配置项
            else:
                data_list = data

        return data_list, config

    def load_annotation_data(self, annotation_file: str) -> List[Dict]:
        """
        加载JSON格式的标注数据

        Args:
            annotation_file: 标注文件路径

        Returns:
            annotations: 标注列表
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('annotations', [])

    def extract_bscan_data(self, data_list: List[Dict], max_encoding: int = 256) -> np.ndarray:
        """
        从JSON数据中提取B扫描数据

        Args:
            data_list: JSON数据列表
            max_encoding: 最大编码值

        Returns:
            bscan_data: B扫描数据数组 [channels, encodings, features]
                       features: [depth, amplitude, db_value]
        """
        # 初始化B扫描矩阵 [9 channels, max_encoding, 3 features]
        bscan_data = np.zeros((9, max_encoding, 3))

        for item in data_list:
            if 'E' in item and 'B1' in item:
                encoding = item['E'] - 1  # E从1开始，转换为0-based索引
                if encoding >= max_encoding:
                    continue

                b1_data = item['B1']
                if len(b1_data) >= 5:  # 确保有足够的数据
                    channel = int(b1_data[0]) - 1  # 通道从1开始，转换为0-based
                    if 0 <= channel < 9:
                        depth = b1_data[1]
                        amplitude = b1_data[2]
                        # 计算dB值 (假设有计算方法，这里暂时使用amplitude)
                        db_value = b1_data[3] if len(b1_data) > 3 else amplitude

                        bscan_data[channel, encoding] = [depth, amplitude, db_value]

        return bscan_data

    def create_sliding_windows(self, bscan_data: np.ndarray, window_size: int = 64,
                              step: int = 32) -> List[np.ndarray]:
        """
        创建滑动窗口数据样本

        Args:
            bscan_data: B扫描数据 [channels, encodings, features]
            window_size: 窗口大小
            step: 滑动步长

        Returns:
            windows: 窗口数据列表，每个元素为 [channels, window_size, features]
        """
        channels, max_encoding, features = bscan_data.shape
        windows = []

        for start in range(0, max_encoding - window_size + 1, step):
            end = start + window_size
            window_data = bscan_data[:, start:end, :]  # [channels, window_size, features]
            windows.append(window_data)

        return windows

    def extract_damage_regions(self, annotations: List[Dict], bscan_data: np.ndarray) -> List[Tuple]:
        """
        从标注数据中提取伤损区域

        Args:
            annotations: 标注列表
            bscan_data: B扫描数据 [channels, encodings, features]

        Returns:
            damage_regions: 伤损区域列表 [(start_encoding, end_encoding, channel, label), ...]
        """
        damage_regions = []

        for annotation in annotations:
            if annotation.get('type') == '伤损':
                subtype = annotation.get('subtype', {})
                flaw_name = subtype.get('name', '')
                flaw_id = self.flaw_types.get(flaw_name, 0)  # 默认为0（正常）

                points = annotation.get('points', [])
                if points:
                    # 计算区域范围
                    channels = set()
                    encodings = []

                    for point in points:
                        channels.add(point.get('channel', 1))
                        encodings.append(point.get('e', 0))

                    if encodings:
                        min_encoding = min(encodings) - 1  # 转换为0-based
                        max_encoding = max(encodings) - 1
                        channel = list(channels)[0]  # 暂时取第一个通道

                        damage_regions.append((min_encoding, max_encoding, channel, flaw_id))

        return damage_regions

    def generate_labels_for_windows(self, windows: List[np.ndarray],
                                   damage_regions: List[Tuple]) -> List[int]:
        """
        为每个窗口生成标签

        Args:
            windows: 窗口数据列表
            damage_regions: 伤损区域列表

        Returns:
            labels: 标签列表
        """
        labels = []

        for i, window in enumerate(windows):
            # 计算窗口的编码范围 (假设每个窗口对应连续的编码位置)
            window_start = i * 32  # step=32
            window_end = window_start + 64  # window_size=64

            # 检查窗口是否与任何伤损区域重叠
            window_label = 0  # 默认为正常

            for region_start, region_end, region_channel, region_label in damage_regions:
                # 检查是否有重叠
                if (window_start < region_end and window_end > region_start):
                    # 检查通道是否匹配
                    # 暂时简化：如果有重叠就标记为伤损
                    window_label = region_label
                    break

            labels.append(window_label)

        return labels

    def convert_to_csv_format(self, windows: List[np.ndarray], labels: List[int],
                            output_file: str):
        """
        将窗口数据转换为CSV格式（类似EMG数据）

        Args:
            windows: 窗口数据列表
            labels: 标签列表
            output_file: 输出CSV文件路径
        """
        # 为每个窗口创建一行数据
        csv_data = []

        for window, label in zip(windows, labels):
            # 将3D窗口数据展平为1D [channels * window_size * features]
            flat_data = window.flatten()

            # 创建一行数据：展平的特征数据 + 标签
            row = list(flat_data) + [label]
            csv_data.append(row)

        # 创建DataFrame并保存
        num_features = len(csv_data[0]) - 1 if csv_data else 0
        columns = [f'feature_{i}' for i in range(num_features)] + ['label']

        df = pd.DataFrame(csv_data, columns=columns)
        df.to_csv(output_file, index=False)

        print(f"Data converted to CSV format: {output_file}")
        print(f"Number of samples: {len(csv_data)}")
        print(f"Feature dimensions: {num_features}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    def convert_single_file(self, data_file: str, annotation_file: str = None,
                           output_file: str = None, window_size: int = 64, step: int = 32):
        """
        转换单个文件的数据

        Args:
            data_file: 原始数据文件路径
            annotation_file: 标注文件路径（可选）
            output_file: 输出CSV文件路径（可选）
            window_size: 窗口大小
            step: 滑动步长
        """
        # 设置默认输出文件名
        if output_file is None:
            data_path = Path(data_file)
            output_file = str(data_path.parent / f"{data_path.stem}_hdc.csv")

        print(f"Converting file: {data_file}")

        # 1. 加载原始数据
        data_list, config = self.load_json_data(data_file)
        print(f"Loaded {len(data_list)} data points")

        # 2. 加载标注数据（如果有）
        annotations = []
        if annotation_file and Path(annotation_file).exists():
            annotations = self.load_annotation_data(annotation_file)
            print(f"Loaded {len(annotations)} annotations")

        # 3. 提取B扫描数据
        max_encoding = 256  # 可以根据需要调整
        bscan_data = self.extract_bscan_data(data_list, max_encoding)
        print(f"B-scan data shape: {bscan_data.shape}")

        # 4. 创建滑动窗口
        windows = self.create_sliding_windows(bscan_data, window_size, step)
        print(f"Generated {len(windows)} window samples")

        # 5. 生成标签
        damage_regions = self.extract_damage_regions(annotations, bscan_data)
        labels = self.generate_labels_for_windows(windows, damage_regions)

        # 6. 转换为CSV格式
        self.convert_to_csv_format(windows, labels, output_file)

        return output_file

    def batch_convert(self, data_dir: str, output_dir: str, window_size: int = 64, step: int = 32):
        """
        批量转换目录中的所有数据文件

        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
            window_size: 窗口大小
            step: 滑动步长
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 查找所有数据文件
        data_files = []
        for ext in ['*.json', '*.xxc']:
            data_files.extend(data_path.rglob(ext))

        converted_files = []

        for data_file in data_files:
            # 跳过标注文件
            if '标注' in data_file.name:
                continue

            # 查找对应的标注文件
            annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"

            # 设置输出文件名
            output_file = output_path / f"{data_file.stem}_hdc.csv"

            try:
                result_file = self.convert_single_file(
                    str(data_file), str(annotation_file), str(output_file),
                    window_size, step
                )
                converted_files.append(result_file)
            except Exception as e:
                print(f"转换文件 {data_file} 时出错: {e}")

        print(f"Batch conversion completed, processed {len(converted_files)} files")
        return converted_files


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description='超声波数据转换工具')
    parser.add_argument('--data_file', help='单个数据文件路径')
    parser.add_argument('--annotation_file', help='标注文件路径')
    parser.add_argument('--data_dir', help='数据目录路径（批量转换）')
    parser.add_argument('--output_file', help='输出CSV文件路径')
    parser.add_argument('--output_dir', default='hdc_data', help='输出目录路径')
    parser.add_argument('--window_size', type=int, default=64, help='窗口大小')
    parser.add_argument('--step', type=int, default=32, help='滑动步长')

    args = parser.parse_args()

    converter = UltrasonicDataConverter()

    if args.data_file:
        # 转换单个文件
        converter.convert_single_file(
            args.data_file, args.annotation_file, args.output_file,
            args.window_size, args.step
        )
    elif args.data_dir:
        # 批量转换
        converter.batch_convert(args.data_dir, args.output_dir, args.window_size, args.step)
    else:
        print("请指定 --data_file 或 --data_dir 参数")


if __name__ == "__main__":
    main()