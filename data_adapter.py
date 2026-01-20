"""
数据适配器
处理实际的超声波探伤数据格式
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class DataAdapter:
    """数据适配器"""
    
    def __init__(self):
        # 类别映射
        self.class_mapping = {
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
    
    def load_original_json(self, json_file: str) -> Tuple[List[Dict], Dict]:
        """加载原始JSON文件
        
        Args:
            json_file: JSON文件路径
        
        Returns:
            (数据列表, BeamOffset配置) 元组
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析BeamOffset配置
        beam_offset = {}
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict) and 'BeamOffset' in first_item:
                beam_offset = first_item['BeamOffset']
        
        if isinstance(data, list):
            return data, beam_offset
        elif isinstance(data, dict):
            return [data], beam_offset
        else:
            return [], beam_offset
    
    def load_annotation_json(self, annotation_file: str) -> Dict:
        """加载标注JSON文件
        
        Args:
            annotation_file: 标注文件路径
        
        Returns:
            标注数据
        """
        if not os.path.exists(annotation_file):
            return {"annotations": []}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def parse_bscan_data(self, data: List[Dict]) -> Dict[int, List[Tuple]]:
        """解析B扫描数据
        
        Args:
            data: 原始数据列表
        
        Returns:
            按编码值分组的B扫描数据
        """
        e_groups = defaultdict(list)
        
        for item in data:
            # 提取编码值
            e_value = item.get('E')
            if e_value is None:
                continue
            
            # 提取所有B扫描数据
            for key, value in item.items():
                if key.startswith('B'):
                    try:
                        # B扫描数据格式: [channel, depth, amplitude, dB_value, ...]
                        bscan_data = value
                        if isinstance(bscan_data, list) and len(bscan_data) >= 3:
                            channel = int(bscan_data[0])
                            depth = float(bscan_data[1])
                            amplitude = float(bscan_data[2])
                            
                            # 添加到分组
                            e_groups[e_value].append((channel, depth, amplitude))
                    except (ValueError, IndexError, TypeError):
                        continue
        
        return dict(e_groups)
    
    def parse_annotations(self, annotation_data: Dict) -> Dict[int, Dict]:
        """解析标注数据
        
        Args:
            annotation_data: 标注数据
        
        Returns:
            按编码值分组的标注数据
        """
        annotations = {}
        
        for ann in annotation_data.get('annotations', []):
            # 提取标注信息
            ann_type = ann.get('type', '')
            subtype = ann.get('subtype', {})
            subtype_name = subtype.get('name', '')
            
            # 获取类别ID
            class_id = self.class_mapping.get(subtype_name, 0)
            
            # 提取点信息
            points = ann.get('points', [])
            for point in points:
                e_value = point.get('e')
                channel = point.get('channel')
                
                if e_value is not None:
                    if e_value not in annotations:
                        annotations[e_value] = {}
                    
                    # 每个通道一个标签
                    annotations[e_value][channel] = class_id
        
        return annotations
    
    def convert_to_csv(self, data: List[Dict], annotation_file: str = None, 
                   output_dir: str = None, beam_offset: Dict = None) -> Tuple[str, str]:
        """将数据转换为CSV格式
        
        Args:
            data: 原始数据
            annotation_file: 标注文件路径
            output_dir: 输出目录
            beam_offset: BeamOffset配置
        
        Returns:
            (data_csv, label_csv) 元组
        """
        # 解析B扫描数据
        bscan_data = self.parse_bscan_data(data)
        
        # 解析标注数据
        annotations = {}
        if annotation_file:
            annotation_data = self.load_annotation_json(annotation_file)
            annotations = self.parse_annotations(annotation_data)
        
        # 生成数据CSV
        data_csv_rows = []
        
        # 按编码值排序
        sorted_e_values = sorted(bscan_data.keys())
        
        for e_value in sorted_e_values:
            # 初始化9个通道的幅度值
            channel_amplitudes = [0.0] * 9
            
            # 填充通道数据
            for channel, depth, amplitude in bscan_data[e_value]:
                if 0 <= channel < 9:
                    channel_amplitudes[channel] = amplitude
            
            # 构建数据CSV行: 编码值 + 9个通道的幅度值
            row = [e_value] + channel_amplitudes
            data_csv_rows.append(row)
        
        # 保存数据CSV文件
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(data[0].get('__file__', '')), 'converted')
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(data[0].get('__file__', '')))[0]
        data_csv = os.path.join(output_dir, f'{base_name}_data.csv')
        
        with open(data_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['encoding'] + [f'channel_{i}' for i in range(9)]
            writer.writerow(header)
            # 写入数据
            writer.writerows(data_csv_rows)
        
        # 生成标签CSV
        label_csv_rows = []
        
        for e_value in sorted_e_values:
            # 初始化9个通道的标签
            channel_labels = [0] * 9
            
            # 填充通道标签
            for channel in range(9):
                if e_value in annotations and channel in annotations[e_value]:
                    channel_labels[channel] = annotations[e_value][channel]
            
            # 构建标签CSV行: 编码值 + 9个通道的标签
            row = [e_value] + channel_labels
            label_csv_rows.append(row)
        
        # 保存标签CSV文件
        label_csv = os.path.join(output_dir, f'{base_name}_label.csv')
        
        with open(label_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['encoding'] + [f'label_{i}' for i in range(9)]
            writer.writerow(header)
            # 写入标签
            writer.writerows(label_csv_rows)
        
        # 保存BeamOffset配置
        if beam_offset:
            beam_offset_file = os.path.join(output_dir, f'{base_name}_beam_offset.json')
            with open(beam_offset_file, 'w', encoding='utf-8') as f:
                json.dump(beam_offset, f, indent=2, ensure_ascii=False)
        
        return data_csv, label_csv
    
    def load_csv(self, data_csv: str, label_csv: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载CSV文件
        
        Args:
            data_csv: 数据CSV文件路径
            label_csv: 标签CSV文件路径
        
        Returns:
            (data, labels) 元组
        """
        data = []
        labels = []
        
        # 加载数据CSV
        with open(data_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 提取编码值和9个通道的幅度值
                encoding = float(row['encoding'])
                channel_amplitudes = [float(row[f'channel_{i}']) for i in range(9)]
                
                # 构建样本: [编码值, 9个幅度值]
                sample = [encoding] + channel_amplitudes
                data.append(sample)
        
        # 加载标签CSV
        with open(label_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 提取编码值和9个通道的标签
                encoding = float(row['encoding'])
                channel_labels = [int(row[f'label_{i}']) for i in range(9)]
                
                # 使用第一个非零标签作为该样本的标签
                non_zero_labels = [label for label in channel_labels if label != 0]
                label = non_zero_labels[0] if non_zero_labels else 0
                labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def merge_csv_files(self, data_csv_files: List[str], label_csv_files: List[str], 
                       output_file: str = None) -> Tuple[str, str]:
        """合并多个CSV文件
        
        Args:
            data_csv_files: 数据CSV文件列表
            label_csv_files: 标签CSV文件列表
            output_file: 输出文件路径（不包含扩展名）
        
        Returns:
            (merged_data_csv, merged_label_csv) 元组
        """
        if output_file is None:
            output_dir = os.path.dirname(data_csv_files[0])
            base_name = os.path.splitext(os.path.basename(data_csv_files[0]))[0]
        else:
            output_dir = os.path.dirname(output_file)
            base_name = os.path.splitext(os.path.basename(output_file))[0]
        
        # 合并数据CSV
        all_data = []
        for csv_file in data_csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    encoding = float(row['encoding'])
                    channel_amplitudes = [float(row[f'channel_{i}']) for i in range(9)]
                    sample = [encoding] + channel_amplitudes
                    all_data.append(sample)
        
        # 保存合并后的数据CSV
        merged_data_csv = os.path.join(output_dir, f'{base_name}_data.csv')
        with open(merged_data_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['encoding'] + [f'channel_{i}' for i in range(9)]
            writer.writerow(header)
            for row in all_data:
                writer.writerow(row)
        
        # 合并标签CSV
        all_labels = []
        for csv_file in label_csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    encoding = float(row['encoding'])
                    channel_labels = [int(row[f'label_{i}']) for i in range(9)]
                    sample = [encoding] + channel_labels
                    all_labels.append(sample)
        
        # 保存合并后的标签CSV
        merged_label_csv = os.path.join(output_dir, f'{base_name}_label.csv')
        with open(merged_label_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['encoding'] + [f'label_{i}' for i in range(9)]
            writer.writerow(header)
            for row in all_labels:
                writer.writerow(row)
        
        print(f"已合并 {len(data_csv_files)} 个数据文件到: {merged_data_csv}")
        print(f"已合并 {len(label_csv_files)} 个标签文件到: {merged_label_csv}")
        
        return merged_data_csv, merged_label_csv
    
    def convert_directory(self, input_dir: str = None, output_dir: str = None) -> List[Tuple[str, str]]:
        """批量转换目录中的文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        
        Returns:
            转换后的CSV文件列表 [(data_csv, label_csv), ...]
        """
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'converted')
        
        os.makedirs(output_dir, exist_ok=True)
        
        csv_file_pairs = []
        
        # 遍历目录中的所有JSON文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.json') and not filename.endswith('标注.json'):
                json_file = os.path.join(input_dir, filename)
                
                # 查找对应的标注文件
                base_name = os.path.splitext(filename)[0]
                annotation_file = os.path.join(input_dir, f'{base_name}标注.json')
                
                # 加载数据和BeamOffset
                data, beam_offset = self.load_original_json(json_file)
                
                # 转换文件
                try:
                    data_csv, label_csv = self.convert_to_csv(
                        data,
                        annotation_file if os.path.exists(annotation_file) else None,
                        output_dir,
                        beam_offset
                    )
                    csv_file_pairs.append((data_csv, label_csv))
                    print(f"已转换: {json_file} -> {data_csv}, {label_csv}")
                except Exception as e:
                    print(f"转换失败 {json_file}: {e}")
        
        return csv_file_pairs