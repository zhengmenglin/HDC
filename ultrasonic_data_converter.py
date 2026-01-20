"""
超声波数据转换器
将JSON格式的原始数据转换为CSV格式
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Optional
import numpy as np
from config import config

class UltrasonicDataConverter:
    """超声波数据转换器"""
    
    def __init__(self, input_dir: str = None, output_dir: str = None):
        self.input_dir = input_dir or config.get("data.input_dir", "data")
        self.output_dir = output_dir or config.get("data.output_dir", "output")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        # 反向映射
        self.class_reverse_mapping = {v: k for k, v in self.class_mapping.items()}
    
    def load_json(self, json_file: str) -> Dict:
        """加载JSON文件"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_annotations(self, annotation_file: str) -> Dict:
        """加载标注文件"""
        if not os.path.exists(annotation_file):
            return {}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = {}
        for ann in data.get("annotations", []):
            ann_type = ann.get("type", "")
            subtype = ann.get("subtype", {})
            subtype_name = subtype.get("name", "")
            
            for point in ann.get("points", []):
                e_value = point.get("e")
                if e_value is not None:
                    annotations[e_value] = self.class_mapping.get(subtype_name, 0)
        
        return annotations
    
    def parse_bscan_data(self, data: Dict) -> List[Tuple]:
        """解析B扫描数据"""
        rows = []
        
        # 遍历所有B扫描数据
        for key, value in data.items():
            if key.startswith("B"):
                try:
                    # 解析B扫描数据: {"B1":[channel,depth,amplitude,...]}
                    bscan_data = value
                    if isinstance(bscan_data, list) and len(bscan_data) >= 2:
                        channel = int(bscan_data[0])
                        depth_values = bscan_data[1:]
                        
                        # 每个深度值作为一个样本
                        for depth in depth_values:
                            if depth is not None:
                                rows.append((key, channel, depth))
                except (ValueError, IndexError, TypeError):
                    continue
        
        return rows
    
    def convert_json_to_csv(self, json_file: str, annotation_file: str = None) -> str:
        """将JSON文件转换为CSV文件"""
        data = self.load_json(json_file)
        annotations = self.load_annotations(annotation_file) if annotation_file else {}
        
        # 解析B扫描数据
        bscan_rows = self.parse_bscan_data(data)
        
        # 按编码值分组
        e_groups = {}
        for key, channel, depth in bscan_rows:
            e_value = int(key[1:])  # 提取编码值
            if e_value not in e_groups:
                e_groups[e_value] = {}
            if channel not in e_groups[e_value]:
                e_groups[e_value][channel] = []
            e_groups[e_value][channel].append(depth)
        
        # 生成CSV数据
        csv_rows = []
        for e_value in sorted(e_groups.keys()):
            channels_data = e_groups[e_value]
            
            # 获取该编码值对应的类标
            label = annotations.get(e_value, 0)
            
            # 构建9个通道的深度值
            row = [e_value]
            for channel in range(9):
                depths = channels_data.get(channel, [])
                if depths:
                    # 取第一个深度值
                    row.append(depths[0])
                else:
                    row.append(0.0)
            
            # 添加类标
            row.append(label)
            csv_rows.append(row)
        
        # 保存CSV文件
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        csv_file = os.path.join(self.output_dir, f"{base_name}.csv")
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['encoding'] + [f'channel_{i}' for i in range(9)] + ['label']
            writer.writerow(header)
            # 写入数据
            writer.writerows(csv_rows)
        
        return csv_file
    
    def convert_directory(self, input_dir: str = None, output_dir: str = None) -> List[str]:
        """批量转换目录中的所有JSON文件"""
        input_dir = input_dir or self.input_dir
        output_dir = output_dir or self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        csv_files = []
        
        # 遍历目录中的所有JSON文件
        for filename in os.listdir(input_dir):
            if filename.endswith('.json'):
                json_file = os.path.join(input_dir, filename)
                
                # 查找对应的标注文件
                base_name = os.path.splitext(filename)[0]
                annotation_file = os.path.join(input_dir, f"{base_name}标注.json")
                
                # 转换文件
                csv_file = self.convert_json_to_csv(json_file, annotation_file)
                csv_files.append(csv_file)
                print(f"已转换: {json_file} -> {csv_file}")
        
        return csv_files
    
    def load_csv(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载CSV文件"""
        data = []
        labels = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 提取特征: 编码值 + 9个通道的深度值
                features = [float(row['encoding'])] + [float(row[f'channel_{i}']) for i in range(9)]
                data.append(features)
                labels.append(int(row['label']))
        
        return np.array(data), np.array(labels)
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """数据归一化"""
        # 对每个特征列进行归一化
        normalized_data = data.copy()
        
        for i in range(data.shape[1]):
            col = data[:, i]
            col_min = np.min(col)
            col_max = np.max(col)
            
            if col_max > col_min:
                normalized_data[:, i] = (col - col_min) / (col_max - col_min)
            else:
                normalized_data[:, i] = 0.0
        
        return normalized_data
    
    def apply_window_sampling(self, data: np.ndarray, labels: np.ndarray, 
                          window_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """应用滑动窗口采样"""
        if len(data) < window_size:
            return data, labels
        
        windowed_data = []
        windowed_labels = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            label = labels[i+window_size//2]  # 使用窗口中心的标签
            windowed_data.append(window.flatten())
            windowed_labels.append(label)
        
        return np.array(windowed_data), np.array(windowed_labels)
    
    def split_train_test(self, data: np.ndarray, labels: np.ndarray, 
                       train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        """划分训练集和测试集"""
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        data = data[indices]
        labels = labels[indices]
        
        # 划分数据
        train_data = data[:n_train]
        train_labels = labels[:n_train]
        test_data = data[n_train:]
        test_labels = labels[n_train:]
        
        return train_data, train_labels, test_data, test_labels