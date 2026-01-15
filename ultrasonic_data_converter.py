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
            "螺孔裂纹": 11,
            "接头": 12,
            "螺孔": 13,
            "导线孔": 14,
            "焊缝": 15,
            "鱼鳞伤": 16,
            "轨底锈蚀": 17,
            "轨底焊筋": 18,
            "道岔": 19
        }

        # 正常结构类型映射 (正常为0)
        self.normal_types = {
            "正常": 0
        }

        # 通道直接使用0-8，不再映射
        
        # 拼孔值映射 (通道到拼孔值的映射)
        self.beam_offset_map = {}

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
                
                # 提取拼孔值（BeamOffset）并创建通道到拼孔值的映射
                if 'BeamOffset' in config_item and isinstance(config_item['BeamOffset'], list):
                    beam_offset = config_item['BeamOffset']
                    # BeamOffset格式：[通道号, 拼孔值, 通道号, 拼孔值, ...]
                    for i in range(0, len(beam_offset), 2):
                        if i + 1 < len(beam_offset):
                            channel = int(beam_offset[i])
                            offset = beam_offset[i + 1]
                            self.beam_offset_map[channel] = offset
                    print(f"Loaded beam offset map: {self.beam_offset_map}")
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

    def extract_bscan_data(self, data_list: List[Dict]) -> Dict[int, Dict[int, List]]:
        """
        从JSON数据中提取B扫描数据，按E值组织

        Args:
            data_list: JSON数据列表

        Returns:
            e_data: 按E值组织的数据，格式为 {e值: {通道: [深度, 振幅, db值], ...}, ...}
        """
        e_data = {}

        for item in data_list:
            if 'E' in item:
                e = item['E']
                if e not in e_data:
                    e_data[e] = {}

                # 处理B1数据，支持同一E值对应多个B1
                if 'B1' in item:
                    b1_data = item['B1']
                    # 确保b1_data是列表
                    if not isinstance(b1_data, list):
                        # 如果是浮点数或其他非列表类型，跳过
                        continue
                    
                    # 检查b1_data是否是嵌套列表
                    if b1_data and isinstance(b1_data[0], list):
                        # 情况1：嵌套列表，每个子列表是一个B1数据项
                        for b_item in b1_data:
                            if len(b_item) >= 5:  # 确保有足够的数据
                                channel = int(b_item[0])  # 通道直接使用原始值
                                if 0 <= channel <= 8:  # 通道范围0-8
                                    depth = b_item[1]
                                    amplitude = b_item[2]
                                    # 计算dB值 (假设有计算方法，这里暂时使用amplitude)
                                    db_value = b_item[3] if len(b_item) > 3 else amplitude

                                    e_data[e][channel] = [depth, amplitude, db_value]
                    elif len(b1_data) >= 5:  # 扁平列表，每5个元素是一个B1数据项
                        # 情况2：扁平列表，包含多个B1数据项
                        # 例如: [2.0,136.0,100.0,257.0,176.0,8.0,126.0,100.0,257.0,176.0] 代表两个B1数据项
                        for i in range(0, len(b1_data), 5):
                            if i + 4 < len(b1_data):  # 确保有足够的数据（i到i+4共5个元素）
                                channel = int(b1_data[i])  # 通道直接使用原始值
                                if 0 <= channel <= 8:  # 通道范围0-8
                                    depth = b1_data[i+1]
                                    amplitude = b1_data[i+2]
                                    # 计算dB值 (假设有计算方法，这里暂时使用amplitude)
                                    db_value = b1_data[i+3] if len(b1_data) > i+3 else amplitude

                                    e_data[e][channel] = [depth, amplitude, db_value]

        return e_data

    def extract_damage_regions(self, annotations: List[Dict]) -> Dict[int, int]:
        """
        从标注数据中提取伤损区域，按E值映射到标签

        Args:
            annotations: 标注列表

        Returns:
            e_labels: E值到标签的映射 {e值: 标签, ...}
        """
        e_labels = {}

        for annotation in annotations:
            if annotation.get('type') in ['伤损', '结构']:
                # 确定类型名称和对应的ID映射
                if annotation.get('type') == '伤损':
                    subtype = annotation.get('subtype', {})
                    type_name = subtype.get('name', '')
                    id_mapping = self.flaw_types
                else:
                    subtype = annotation.get('subtype', {})
                    type_name = subtype.get('name', '')
                    id_mapping = self.flaw_types  # 结构类型也使用伤损类型映射

                label_id = id_mapping.get(type_name, 0)  # 默认为0（正常）

                points = annotation.get('points', [])
                for point in points:
                    e = point.get('e', 0)
                    if e > 0:
                        e_labels[e] = label_id

        return e_labels

    def convert_to_csv_format(self, e_data: Dict[int, Dict[int, List]], e_labels: Dict[int, int],
                            output_file: str):
        """
        将E值数据转换为CSV格式
        输出格式：编码值 + 9通道深度值 + 类别

        Args:
            e_data: 按E值组织的数据 {e值: {通道: [深度, 振幅, db值], ...}, ...}
            e_labels: E值到标签的映射 {e值: 标签, ...}
            output_file: 输出CSV文件路径
        """
        # 为每个E值创建一行数据
        csv_data = []

        # 按E值排序
        sorted_es = sorted(e_data.keys())

        for e in sorted_es:
            # 获取当前E值的通道数据
            channel_data = e_data[e]
            
            # 初始化行数据：编码值 + 9通道深度值（初始为0） + 类别
            # 使用原始E值作为编码值，不再计算最小编码值
            encoding = e
            
            row = [encoding] + [0.0] * 9 + [0]
            
            # 填充各通道的深度值
            for channel in range(9):  # 通道0-8
                if channel in channel_data:
                    row[channel+1] = channel_data[channel][0]  # 第0个元素是深度值，row[0]是编码值，通道数据从row[1]开始
            
            # 设置类别
            row[-1] = e_labels.get(e, 0)  # 默认为0（正常）
            
            csv_data.append(row)

        # 创建DataFrame并保存
        columns = ['encoding'] + [f'channel_{i}' for i in range(9)] + ['label']

        df = pd.DataFrame(csv_data, columns=columns)
        df.to_csv(output_file, index=False)

        print(f"Data converted to CSV format: {output_file}")
        print(f"Number of samples: {len(csv_data)}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    def convert_windows_to_csv_format(self, windows: List[Tuple[np.ndarray, int]], output_file: str):
        """
        将窗口数据转换为CSV格式

        Args:
            windows: 窗口数据列表，每个元素为 (window_data, label)
                    window_data: [channels, window_size, features] 格式
            output_file: 输出CSV文件路径
        """
        csv_data = []

        for window_data, label in windows:
            # 将窗口数据展平为特征向量
            # window_data形状: [channels, window_size, features]
            # 展平后: [channels * window_size * features]
            features = window_data.flatten()
            
            # 创建一行数据
            row = list(features) + [label]
            csv_data.append(row)

        # 计算特征数量
        num_features = len(csv_data[0]) - 1 if csv_data else 0
        columns = [f'feature_{i}' for i in range(num_features)] + ['label']

        df = pd.DataFrame(csv_data, columns=columns)
        df.to_csv(output_file, index=False)

        print(f"Data converted to CSV format: {output_file}")
        print(f"Number of samples: {len(csv_data)}")
        print(f"Feature dimensions: {num_features}")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    def create_sliding_windows(self, e_data: Dict[int, Dict[int, List]], e_labels: Dict[int, int], window_size: int = 64, step: int = 32) -> List[Tuple[np.ndarray, int]]:
        """
        创建滑动窗口数据样本

        Args:
            e_data: 按E值组织的数据 {e值: {通道: [深度, 振幅, db值], ...}, ...}
            window_size: 窗口大小
            step: 滑动步长

        Returns:
            windows: 窗口数据列表，每个元素为 (window_data, label)
                    window_data: [channels, window_size, 3] 格式
        """
        windows = []

        # 获取所有E值并排序
        sorted_es = sorted(e_data.keys())
        total_es = len(sorted_es)

        for start_idx in range(0, total_es - window_size + 1, step):
            end_idx = start_idx + window_size
            window_es = sorted_es[start_idx:end_idx]

            # 初始化窗口数据 [channels, window_size, 3]
            window_data = np.zeros((9, window_size, 3))

            # 填充窗口数据
            for win_pos, e in enumerate(window_es):
                channel_data = e_data[e]
                
                for channel in range(9):  # 通道0-8
                    if channel in channel_data:
                        window_data[channel, win_pos] = channel_data[channel]

            # 确定窗口标签（使用窗口内出现最多的标签）
            window_labels = []
            for e in window_es:
                if e in e_labels:
                    window_labels.append(e_labels[e])

            if window_labels:
                # 使用出现最多的标签
                label = max(set(window_labels), key=window_labels.count)
            else:
                label = 0  # 默认为正常

            windows.append((window_data, label))

        return windows

    def convert_single_file(self, data_file: str, annotation_file: str = None,
                           output_file: str = None, window_size: int = 64, step: int = 32):
        """
        转换单个文件的数据

        Args:
            data_file: 原始数据文件路径
            annotation_file: 标注文件路径（可选）
            output_file: 输出CSV文件路径（可选）
            window_size: 窗口大小（保留参数，兼容旧接口）
            step: 滑动步长（保留参数，兼容旧接口）
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

        # 3. 提取B扫描数据（按E值组织）
        e_data = self.extract_bscan_data(data_list)
        print(f"Extracted data for {len(e_data)} E values")

        # 4. 生成E值到标签的映射
        e_labels = self.extract_damage_regions(annotations)
        print(f"Labeled {len(e_labels)} E values with damage information")

        # 5. 转换为CSV格式（直接生成原始格式，由模型训练时生成滑动窗口）
        self.convert_to_csv_format(e_data, e_labels, output_file)

        return output_file

    def batch_convert(self, data_dir: str, output_dir: str):
        """
        批量转换目录中的所有数据文件

        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
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
                    str(data_file), str(annotation_file), str(output_file)
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

    args = parser.parse_args()

    converter = UltrasonicDataConverter()

    if args.data_file:
        # 转换单个文件
        converter.convert_single_file(
            args.data_file, args.annotation_file, args.output_file
        )
    elif args.data_dir:
        # 批量转换
        converter.batch_convert(args.data_dir, args.output_dir)
    else:
        print("请指定 --data_file 或 --data_dir 参数")


if __name__ == "__main__":
    main()