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
                            if len(b_item) >= 3:  # 至少需要通道、深度、幅度
                                channel = int(b_item[0])  # 通道直接使用原始值
                                if 0 <= channel <= 8:  # 通道范围0-8
                                    depth = b_item[1]
                                    amplitude = b_item[2]
                                    # 计算dB值 (假设有计算方法，这里暂时使用amplitude)
                                    db_value = b_item[3] if len(b_item) > 3 else amplitude

                                    e_data[e][channel] = [depth, amplitude, db_value]
                    elif len(b1_data) >= 3:  # 扁平列表，每5个元素是一个B1数据项
                        # 情况2：扁平列表，包含多个B1数据项
                        # 例如: [2.0,136.0,100.0,257.0,176.0,8.0,126.0,100.0,257.0,176.0] 代表两个B1数据项
                        for i in range(0, len(b1_data), 5):
                            if i + 2 < len(b1_data):  # 确保有足够的数据（至少通道、深度、幅度）
                                channel = int(b1_data[i])  # 通道直接使用原始值
                                if 0 <= channel <= 8:  # 通道范围0-8
                                    depth = b1_data[i+1]
                                    amplitude = b1_data[i+2]
                                    # 计算dB值 (假设有计算方法，这里暂时使用amplitude)
                                    db_value = b1_data[i+3] if i + 3 < len(b1_data) else amplitude

                                    e_data[e][channel] = [depth, amplitude, db_value]

        return e_data

    def extract_damage_regions(self, annotations: List[Dict]) -> Dict[int, Dict[int, int]]:
        """
        从标注数据中提取伤损区域，按E值和通道映射到标签

        Args:
            annotations: 标注列表

        Returns:
            e_channel_labels: E值到通道标签的映射 {e值: {通道: 标签, ...}, ...}
        """
        e_channel_labels = {}

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
                        channel = point.get('channel', 0)
                        if e not in e_channel_labels:
                            e_channel_labels[e] = {}
                        e_channel_labels[e][channel] = label_id

        return e_channel_labels


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

    def generate_all_csvs(self, json_file: str, annotation_file: str = None):
        """
        按照 suanfaliucheng.md #L50-57 的方案生成四个CSV文件
        1. 1.csv: original_e, channel_0-8 (声程值)
        2. 1_encoding.csv: original_e, encoding_0-8 (调整后的编码值)
        3. 1_hdc.csv: encoding, channel_0-8 (对齐后的声程值)
        4. 1_label.csv: encoding, label_0-8 (标注标签)
        """
        json_path = Path(json_file)
        base_name = json_path.stem
        parent_dir = json_path.parent

        # 1. 读取1.json文件 (只读取一次)
        data_list, config = self.load_json_data(json_file)
        beam_offset = self.beam_offset_map
        
        # 提取B扫描数据（按E值组织）
        # e_data[e][channel] = [depth, amplitude, db_value]
        e_data = self.extract_bscan_data(data_list)
        sorted_es = sorted(e_data.keys())
        
        # --- Step 1: 生成 1.csv ---
        # 包含 original_e 和各通道的声程值 (depth)
        csv1_data = []
        for e in sorted_es:
            row = [e] + [0.0] * 9
            for channel, values in e_data[e].items():
                if 0 <= channel <= 8:
                    row[channel + 1] = values[0]  # depth/声程值
            csv1_data.append(row)
        
        df1 = pd.DataFrame(csv1_data, columns=['original_e'] + [f'channel_{i}' for i in range(9)])
        csv1_path = parent_dir / f"{base_name}.csv"
        df1.to_csv(csv1_path, index=False)
        print(f"Generated {csv1_path}")

        # --- Step 2: 生成 1_encoding.csv ---
        # 调整 original_e 得到 encoding_0-8
        # encoding = original_e - beam_offset[channel]
        csv_enc_data = []
        all_encodings = []
        for e in sorted_es:
            row = [e] + [0] * 9
            for channel in range(9):
                offset = beam_offset.get(channel, 0)
                enc = e - offset
                row[channel + 1] = enc
                all_encodings.append(enc)
            csv_enc_data.append(row)
        
        df_enc = pd.DataFrame(csv_enc_data, columns=['original_e'] + [f'encoding_{i}' for i in range(9)])
        csv_enc_path = parent_dir / f"{base_name}_encoding.csv"
        df_enc.to_csv(csv_enc_path, index=False)
        print(f"Generated {csv_enc_path}")

        # --- Step 3: 生成 1_hdc.csv ---
        # 找到 encoding 的范围
        if not all_encodings:
            print("No encodings found, skipping hdc and label csv.")
            return

        min_e = min(all_encodings)
        max_e = max(all_encodings)
        encoding_range = range(int(min_e), int(max_e) + 1)

        csv_hdc_data = []
        for enc in encoding_range:
            row = [enc] + [0.0] * 9
            for channel in range(9):
                offset = beam_offset.get(channel, 0)
                original_e = enc + offset
                if original_e in e_data and channel in e_data[original_e]:
                    row[channel + 1] = e_data[original_e][channel][0] # depth
            csv_hdc_data.append(row)
        
        df_hdc = pd.DataFrame(csv_hdc_data, columns=['encoding'] + [f'channel_{i}' for i in range(9)])
        csv_hdc_path = parent_dir / f"{base_name}_hdc.csv"
        df_hdc.to_csv(csv_hdc_path, index=False)
        print(f"Generated {csv_hdc_path}")

        # --- Step 4: 生成 1_label.csv ---
        # 读取标注文件
        csv_label_data = []
        e_channel_labels = {}
        if annotation_file and Path(annotation_file).exists():
            annotations = self.load_annotation_data(annotation_file)
            e_channel_labels = self.extract_damage_regions(annotations)
        
        for enc in encoding_range:
            row = [enc] + [0] * 9
            for channel in range(9):
                offset = beam_offset.get(channel, 0)
                original_e = enc + offset
                if original_e in e_channel_labels and channel in e_channel_labels[original_e]:
                    row[channel + 1] = e_channel_labels[original_e][channel]
            csv_label_data.append(row)
        
        df_label = pd.DataFrame(csv_label_data, columns=['encoding'] + [f'label_{i}' for i in range(9)])
        csv_label_path = parent_dir / f"{base_name}_label.csv"
        df_label.to_csv(csv_label_path, index=False)
        print(f"Generated {csv_label_path}")

    def convert_to_csv_with_beam_offset(self, json_file: str, output_file: str = None):
        """
        根据新方案将JSON数据转换为CSV格式
        1. 读取JSON文件，获取所有通道的编码值和声程值
        2. 将channel_0-8共9列数据分别上移对应的BeamOffset行
        3. 空缺位置填充为0，E按照规律填充

        Args:
            json_file: JSON文件路径
            output_file: 输出CSV文件路径

        Returns:
            output_file: 输出文件路径
        """
        # 设置默认输出文件名
        if output_file is None:
            data_path = Path(json_file)
            output_file = str(data_path.parent / f"{data_path.stem}.csv")

        print(f"Converting file with new beam offset method: {json_file}")

        # 1. 加载JSON数据
        data_list, config = self.load_json_data(json_file)
        print(f"Loaded {len(data_list)} data points")

        # 2. 提取B扫描数据（按E值组织）
        e_data = self.extract_bscan_data(data_list)
        print(f"Extracted data for {len(e_data)} E values")

        # 3. 获取所有E值并排序
        sorted_es = sorted(e_data.keys())
        if not sorted_es:
            print("No E values found in data")
            return output_file

        min_e = min(sorted_es)
        max_e = max(sorted_es)

        # 4. 应用BeamOffset调整（数据上移）
        beam_offset = self.beam_offset_map
        print(f"Beam offset map: {beam_offset}")

        # 确定调整后的编码范围（考虑最大偏移）
        max_offset = max(beam_offset.values()) if beam_offset else 0
        adjusted_min_e = min_e - max_offset
        adjusted_max_e = max_e

        # 创建调整后的编码范围
        adjusted_encoding_range = range(adjusted_min_e, adjusted_max_e + 1)

        # 填充数据（应用BeamOffset调整）
        csv_data = []
        for encoding in adjusted_encoding_range:
            row = [encoding] + [0.0] * 9  # encoding + channel_0-8

            for channel in range(9):
                offset = beam_offset.get(channel, 0)
                original_encoding = encoding + offset

                # 如果原始编码在数据范围内且有数据，则填充
                if original_encoding in e_data and channel in e_data[original_encoding]:
                    depth_value = e_data[original_encoding][channel][0]
                    row[channel + 1] = depth_value

            csv_data.append(row)

        # 创建DataFrame
        df = pd.DataFrame(csv_data, columns=['encoding'] + [f'channel_{i}' for i in range(9)])
        print(f"DataFrame created with {len(df)} rows")

        # 5. 保存为CSV文件
        df.to_csv(output_file, index=False)

        print(f"Data converted to CSV format: {output_file}")
        print(f"Number of samples: {len(df)}")
        print(f"Encoding range: {adjusted_min_e} to {adjusted_max_e}")

        return output_file

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
            output_file = str(data_path.parent / f"{data_path.stem}.csv")

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

        # 4. 如果有标注数据，生成E值到标签的映射
        e_labels = {}
        if annotations:
            e_labels = self.extract_damage_regions(annotations)
            print(f"Labeled {len(e_labels)} E values with damage information")

        # 5. 应用BeamOffset调整并转换为CSV格式
        sorted_es = sorted(e_data.keys())
        if not sorted_es:
            print("No E values found in data")
            return output_file

        min_e = min(sorted_es)
        max_e = max(sorted_es)

        # 应用BeamOffset调整
        beam_offset = self.beam_offset_map
        print(f"Beam offset map: {beam_offset}")

        # 确定调整后的编码范围（考虑最大偏移）
        max_offset = max(beam_offset.values()) if beam_offset else 0
        adjusted_min_e = min_e - max_offset
        adjusted_max_e = max_e

        # 创建调整后的编码范围
        adjusted_encoding_range = range(adjusted_min_e, adjusted_max_e + 1)

        # 填充数据（应用BeamOffset调整）
        csv_data = []
        for encoding in adjusted_encoding_range:
            row = [encoding] + [0.0] * 9 + [0] * 9  # encoding + channel_0-8 + label_0-8

            for channel in range(9):
                offset = beam_offset.get(channel, 0)
                original_encoding = encoding + offset

                # 如果原始编码在数据范围内且有数据，则填充深度值
                if original_encoding in e_data and channel in e_data[original_encoding]:
                    depth_value = e_data[original_encoding][channel][0]
                    row[channel + 1] = depth_value  # 填充channel数据

                # 如果有标注数据，填充标签值
                if original_encoding in e_labels and channel in e_labels[original_encoding]:
                    row[10 + channel] = e_labels[original_encoding][channel]  # 填充label数据

            csv_data.append(row)

        # 创建DataFrame
        columns = ['encoding'] + [f'channel_{i}' for i in range(9)] + [f'label_{i}' for i in range(9)]
        df = pd.DataFrame(csv_data, columns=columns)
        print(f"DataFrame created with {len(df)} rows")

        # 6. 保存为CSV文件
        df.to_csv(output_file, index=False)

        print(f"Data converted to CSV format: {output_file}")
        print(f"Number of samples: {len(df)}")
        print(f"Encoding range: {adjusted_min_e} to {adjusted_max_e}")

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
            output_file = output_path / f"{data_file.stem}.csv"

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