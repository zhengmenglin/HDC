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
    
    def load_original_json(self, json_file: str) -> List[Dict]:
        """加载原始JSON文件
        
        Args:
            json_file: JSON文件路径
        
        Returns:
            数据列表
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
    
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
                   output_csv: str = None) -> str:
        """将数据转换为CSV格式
        
        Args:
            data: 原始数据
            annotation_file: 标注文件路径
            output_csv: 输出CSV文件路径
        
        Returns:
            CSV文件路径
        """
        # 解析B扫描数据
        bscan_data = self.parse_bscan_data(data)
        
        # 解析标注数据
        annotations = {}
        if annotation_file:
            annotation_data = self.load_annotation_json(annotation_file)
            annotations = self.parse_annotations(annotation_data)
        
        # 生成CSV数据
        csv_rows = []
        
        # 按编码值排序
        sorted_e_values = sorted(bscan_data.keys())
        
        for e_value in sorted_e_values:
            # 初始化9个通道的深度值
            channel_depths = [0.0] * 9
            channel_labels = [0] * 9
            
            # 填充通道数据
            for channel, depth, amplitude in bscan_data[e_value]:
                if 0 <= channel < 9:
                    channel_depths[channel] = depth
                    
                    # 获取标签
                    if e_value in annotations and channel in annotations[e_value]:
                        channel_labels[channel] = annotations[e_value][channel]
            
            # 构建CSV行: 编码值 + 9个通道的深度值 + 9个标签
            row = [e_value] + channel_depths + channel_labels
            csv_rows.append(row)
        
        # 保存CSV文件
        if output_csv is None:
            output_csv = os.path.splitext(data_file)[0] + '_converted.csv'
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['encoding'] + [f'channel_{i}' for i in range(9)] + [f'label_{i}' for i in range(9)]
            writer.writerow(header)
            # 写入数据
            writer.writerows(csv_rows)
        
        return output_csv
    
    def convert_directory(self, input_dir: str, output_dir: str = None) -> List[str]:
        """批量转换目录中的文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        
        Returns:
            转换后的CSV文件列表
        """
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'converted')
        
        os.makedirs(output_dir, exist_ok=True)
        
        csv_files = []
        
        # 遍历目录中的所有JSON文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.json') and not filename.endswith('标注.json'):
                json_file = os.path.join(input_dir, filename)
                
                # 查找对应的标注文件
                base_name = os.path.splitext(filename)[0]
                annotation_file = os.path.join(input_dir, f'{base_name}标注.json')
                
                # 转换文件
                output_csv = os.path.join(output_dir, f'{base_name}_converted.csv')
                try:
                    csv_file = self.convert_to_csv(
                        self.load_original_json(json_file),
                        annotation_file if os.path.exists(annotation_file) else None,
                        output_csv
                    )
                    csv_files.append(csv_file)
                    print(f"已转换: {json_file} -> {csv_file}")
                except Exception as e:
                    print(f"转换失败 {json_file}: {e}")
        
        return csv_files
    
    def load_csv(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载CSV文件
        
        Args:
            csv_file: CSV文件路径
        
        Returns:
            (data, labels) 元组
        """
        data = []
        labels = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 提取编码值和9个通道的深度值
                encoding = float(row['encoding'])
                channel_depths = [float(row[f'channel_{i}']) for i in range(9)]
                
                # 提取9个标签
                channel_labels = [int(row[f'label_{i}']) for i in range(9)]
                
                # 构建样本: [编码值, 9个深度值]
                sample = [encoding] + channel_depths
                data.append(sample)
                
                # 使用第一个非零标签作为该样本的标签
                non_zero_labels = [label for label in channel_labels if label != 0]
                label = non_zero_labels[0] if non_zero_labels else 0
                labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def merge_csv_files(self, csv_files: List[str], output_file: str) -> str:
        """合并多个CSV文件
        
        Args:
            csv_files: CSV文件列表
            output_file: 输出文件路径
        
        Returns:
            合并后的CSV文件路径
        """
        all_data = []
        all_labels = []
        
        for csv_file in csv_files:
            data, labels = self.load_csv(csv_file)
            all_data.append(data)
            all_labels.append(labels)
        
        # 合并数据
        merged_data = np.concatenate(all_data, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        
        # 保存合并后的CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['encoding'] + [f'channel_{i}' for i in range(9)] + ['label']
            writer.writerow(header)
            # 写入数据
            for i in range(len(merged_data)):
                row = merged_data[i].tolist() + [int(merged_labels[i])]
                writer.writerow(row)
        
        print(f"已合并 {len(csv_files)} 个文件到: {output_file}")
        return output_file
    
    def create_train_test_split(self, csv_files: List[str], 
                            train_ratio: float = 0.7,
                            output_dir: str = None) -> Tuple[str, str]:
        """创建训练集和测试集
        
        Args:
            csv_files: CSV文件列表
            train_ratio: 训练集比例
            output_dir: 输出目录
        
        Returns:
            (train_csv, test_csv) 元组
        """
        if output_dir is None:
            output_dir = os.path.dirname(csv_files[0])
        
        # 合并所有CSV文件
        merged_file = os.path.join(output_dir, 'merged_data.csv')
        self.merge_csv_files(csv_files, merged_file)
        
        # 加载合并后的数据
        data, labels = self.load_csv(merged_file)
        
        # 随机打乱数据
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
        # 划分训练集和测试集
        n_train = int(len(data) * train_ratio)
        
        train_data = data[:n_train]
        train_labels = labels[:n_train]
        test_data = data[n_train:]
        test_labels = labels[n_train:]
        
        # 保存训练集
        train_csv = os.path.join(output_dir, 'train_data.csv')
        with open(train_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['encoding'] + [f'channel_{i}' for i in range(9)] + ['label']
            writer.writerow(header)
            for i in range(len(train_data)):
                row = train_data[i].tolist() + [int(train_labels[i])]
                writer.writerow(row)
        
        # 保存测试集
        test_csv = os.path.join(output_dir, 'test_data.csv')
        with open(test_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['encoding'] + [f'channel_{i}' for i in range(9)] + ['label']
            writer.writerow(header)
            for i in range(len(test_data)):
                row = test_data[i].tolist() + [int(test_labels[i])]
                writer.writerow(row)
        
        print(f"训练集: {train_csv} ({len(train_data)} 样本)")
        print(f"测试集: {test_csv} ({len(test_data)} 样本)")
        
        return train_csv, test_csv