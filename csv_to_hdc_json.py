#!/usr/bin/env python3
"""
将CSV格式的超声波HDC检测结果转换为hdc.json格式
使用DBSCAN聚类算法将同一目标的点归为一个伤损结构
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 导入坐标转换工具
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.coordinate_converter import converter


class CsvToHdcJsonConverter:
    """
    CSV到hdc.json转换器
    使用DBSCAN聚类算法将同一目标的点归为一个伤损结构
    """
    
    def __init__(self):
        # 伤损类型映射
        self.flaw_types = {
            0: "正常",
            1: "轨头外侧核伤",
            2: "轨头中心核伤",
            3: "轨头内侧核伤",
            4: "轨头内、外、中心核伤",
            5: "轨头内、中心核伤",
            6: "轨头外、中心核伤",
            7: "轨头水平分裂",
            8: "轨头、轨腰分裂",
            9: "轨腰水平裂",
            10: "轨底断裂",
            11: "螺孔裂纹",
            12: "接头",
            13: "螺孔",
            14: "导线孔",
            15: "焊缝",
            16: "鱼鳞伤",
            17: "轨底锈蚀",
            18: "轨底焊筋",
            19: "道岔"
        }
    
    def load_csv(self, csv_file: str) -> pd.DataFrame:
        """
        加载CSV文件
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            加载后的DataFrame
        """
        return pd.read_csv(csv_file)
    
    def extract_defect_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取缺陷点（label或predicted_label不为0的点）
        支持单通道标签（label）和多通道标签（label_0到label_8）
        
        Args:
            df: 包含检测结果的DataFrame
            
        Returns:
            缺陷点的DataFrame
        """
        # 检查是否有多通道标签
        has_multichannel_pred = all([f'predicted_label_{i}' in df.columns for i in range(9)])
        has_multichannel_true = all([f'label_{i}' in df.columns for i in range(9)])
        
        if has_multichannel_pred:
            # 多通道预测结果，检查任意通道的预测标签不为0即为缺陷点
            pred_mask = pd.Series([False] * len(df), index=df.index)
            for i in range(9):
                pred_mask = pred_mask | (df[f'predicted_label_{i}'] != 0)
            return df[pred_mask].copy()
        elif has_multichannel_true:
            # 多通道真实标签，检查任意通道的真实标签不为0即为缺陷点
            true_mask = pd.Series([False] * len(df), index=df.index)
            for i in range(9):
                true_mask = true_mask | (df[f'label_{i}'] != 0)
            return df[true_mask].copy()
        elif 'predicted_label' in df.columns:
            # 单通道预测结果
            return df[df['predicted_label'] != 0].copy()
        else:
            # 单通道真实标签
            return df[df['label'] != 0].copy()
    
    def cluster_points(self, df: pd.DataFrame, eps: float = 10.0, min_samples: int = 3, 
                      beam_offset_map: Dict[int, float] = None, 
                      beam_way_map: Dict[int, int] = None,
                      beam_angle_map: Dict[int, float] = None) -> pd.DataFrame:
        """
        使用DBSCAN聚类算法将同一目标的点归为一个簇，基于显示坐标
        支持多通道标签聚类，考虑所有通道的预测结果
        
        Args:
            df: 缺陷点DataFrame
            eps: DBSCAN的ε参数，两个样本被视为同一簇的最大距离
            min_samples: DBSCAN的min_samples参数，一个簇的最小样本数
            beam_offset_map: 通道到拼孔值的映射
            beam_way_map: 通道到射向的映射（-1后、0直、1前）
            beam_angle_map: 通道到声束角度的映射
            
        Returns:
            带有簇标签的DataFrame
        """
        if df.empty:
            df['cluster'] = []
            return df
        
        if beam_offset_map is None:
            beam_offset_map = {}
        if beam_way_map is None:
            beam_way_map = {}
        if beam_angle_map is None:
            beam_angle_map = {}
        
        # 检查是否有多通道标签
        has_multichannel_pred = all([f'predicted_label_{i}' in df.columns for i in range(9)])
        has_multichannel_true = all([f'label_{i}' in df.columns for i in range(9)])
        
        # 为每个行生成聚类特征，考虑所有通道的信息
        all_cluster_features = []
        row_indices = []
        channels_used = []
        
        # 遍历每个行，收集所有通道的有效数据
        for idx, row in df.iterrows():
            # 收集当前行所有非零深度值的通道和深度
            valid_channels = []
            for channel in range(9):
                depth = row[f'channel_{channel}']
                if depth != 0:
                    # 检查当前通道是否有预测结果
                    has_pred = has_multichannel_pred and row[f'predicted_label_{channel}'] != 0
                    has_true = has_multichannel_true and row[f'label_{channel}'] != 0
                    
                    if has_pred or has_true:
                        valid_channels.append((channel, depth))
            
            # 为每个有效通道生成聚类特征
            for channel, depth in valid_channels:
                # 使用坐标转换工具计算显示坐标
                beam_offset = beam_offset_map.get(channel, 0)
                beam_way = beam_way_map.get(channel, 0)
                beam_angle = beam_angle_map.get(channel, None)
                
                # 计算显示坐标
                adjusted_e, adjusted_depth = converter.calculate_display_coordinates(
                    e=int(row['encoding']),
                    channel=channel,
                    depth=depth,
                    beam_offset=beam_offset,
                    beam_way=beam_way,
                    beam_angle=beam_angle,
                    display_mode="压缩"
                )
                
                # 使用调整后的坐标作为聚类特征
                all_cluster_features.append([adjusted_e, adjusted_depth, channel])
                row_indices.append(idx)
                channels_used.append(channel)
        
        # 初始化簇标签
        df['cluster'] = -1
        
        if all_cluster_features:
            # 将特征转换为numpy数组
            X = np.array(all_cluster_features)
            
            # 使用DBSCAN聚类，调整参数以适应螺孔的连续结构
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)
            
            # 将簇标签分配回原始DataFrame
            # 注意：同一行可能有多个通道的数据，可能被分到不同簇
            # 这里我们使用最小的簇标签（非噪声点）作为行的簇标签
            for i, (idx, channel, cluster) in enumerate(zip(row_indices, channels_used, clusters)):
                if cluster != -1:  # 非噪声点
                    current_cluster = df.at[idx, 'cluster']
                    if current_cluster == -1 or cluster < current_cluster:
                        df.at[idx, 'cluster'] = cluster
        
        return df
    
    def extract_beam_parameters(self, original_json_path: str) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, float]]:
        """
        从原始JSON文件中提取超声波参数
        
        Args:
            original_json_path: 原始JSON文件路径
            
        Returns:
            beam_offset_map: 通道到拼孔值的映射
            beam_way_map: 通道到射向的映射（-1后、0直、1前）
            beam_angle_map: 通道到声束角度的映射
        """
        beam_offset_map = {}
        beam_way_map = {}
        beam_angle_map = {}
        
        if not os.path.exists(original_json_path):
            return beam_offset_map, beam_way_map, beam_angle_map
        
        try:
            with open(original_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取配置信息（第一个元素通常是配置）
            if data and isinstance(data[0], dict):
                config_item = data[0]
                if 'E' in config_item and isinstance(config_item['E'], int) and config_item['E'] == 1:
                    # 提取BeamOffset
                    if 'BeamOffset' in config_item and isinstance(config_item['BeamOffset'], list):
                        beam_offset = config_item['BeamOffset']
                        # BeamOffset格式：[通道号, 拼孔值, 通道号, 拼孔值, ...]
                        for i in range(0, len(beam_offset), 2):
                            if i + 1 < len(beam_offset):
                                channel = int(beam_offset[i])
                                offset = beam_offset[i + 1]
                                beam_offset_map[channel] = offset
                    
                    # 提取BeamWay
                    if 'BeamWay' in config_item and isinstance(config_item['BeamWay'], list):
                        beam_way = config_item['BeamWay']
                        # BeamWay格式：每三个数为一组，依次是通道号、射向（-1后、0直、1前）、侧边（-1左、0正、1右）
                        for i in range(0, len(beam_way), 3):
                            if i + 2 < len(beam_way):
                                channel = int(beam_way[i])
                                way = int(beam_way[i + 1])
                                beam_way_map[channel] = way
                    
                    # 提取BeamAngle
                    if 'BeamAngle' in config_item and isinstance(config_item['BeamAngle'], list):
                        beam_angle = config_item['BeamAngle']
                        # BeamAngle格式：[通道号, 角度, 通道号, 角度, ...]
                        for i in range(0, len(beam_angle), 2):
                            if i + 1 < len(beam_angle):
                                channel = int(beam_angle[i])
                                angle = float(beam_angle[i + 1])
                                beam_angle_map[channel] = angle
        except Exception as e:
            print(f"提取超声波参数时出错: {e}")
        
        return beam_offset_map, beam_way_map, beam_angle_map
    
    def generate_hdc_json(self, clustered_df: pd.DataFrame, original_csv_path: str) -> Dict:
        """
        生成hdc.json格式的数据
        
        Args:
            clustered_df: 带有簇标签的缺陷点DataFrame
            original_csv_path: 原始CSV文件路径
            
        Returns:
            hdc.json格式的字典
        """
        annotations = []
        cluster_id = 1000  # 自动标注ID从1000开始，减少与手动标注ID冲突的概率
        
        # 检查是否有多通道标签
        has_multichannel_pred = all([f'predicted_label_{i}' in clustered_df.columns for i in range(9)])
        has_multichannel_true = all([f'label_{i}' in clustered_df.columns for i in range(9)])
        
        # 从原始JSON文件中提取超声波参数
        # 构造原始JSON文件路径：将_csv_hdc.csv替换为原始扩展名
        csv_path = Path(original_csv_path)
        # 处理文件名：例如 1_hdc.csv -> 1.json
        original_stem = csv_path.stem.replace('_hdc', '')
        original_json_path = str(csv_path.parent / f"{original_stem}.json")
        # 提取超声波参数映射
        beam_offset_map, beam_way_map, beam_angle_map = self.extract_beam_parameters(original_json_path)
        print(f"Using beam offset map: {beam_offset_map} for {original_csv_path}")
        print(f"Using beam way map: {beam_way_map} for {original_csv_path}")
        print(f"Using beam angle map: {beam_angle_map} for {original_csv_path}")
        
        # 按簇分组
        for cluster in clustered_df['cluster'].unique():
            if cluster == -1:  # 噪声点，跳过
                continue
                
            cluster_points = clustered_df[clustered_df['cluster'] == cluster]
            if cluster_points.empty:
                continue
            
            if has_multichannel_pred or has_multichannel_true:
                # 多通道标签处理
                for channel in range(9):
                    channel_points = cluster_points[cluster_points[f'channel_{channel}'] != 0]
                    if channel_points.empty:
                        continue
                    
                    # 获取该通道的标签
                    if has_multichannel_pred:
                        channel_labels = channel_points[f'predicted_label_{channel}']
                    else:
                        channel_labels = channel_points[f'label_{channel}']
                    
                    # 过滤掉标签为0的点
                    channel_points = channel_points[channel_labels != 0]
                    if channel_points.empty:
                        continue
                    
                    # 确定簇该通道的主要类型（出现次数最多的标签）
                    main_label = channel_labels[channel_labels != 0].mode().iloc[0] if not channel_labels[channel_labels != 0].empty else 0
                    
                    # 收集该通道的所有点
                    points = []
                    for _, row in channel_points.iterrows():
                        # 获取该通道的深度值
                        depth = row[f'channel_{channel}']
                        
                        # 获取该通道的拼孔值
                        offset = beam_offset_map.get(channel, 0)
                        # 逆拼孔处理：原始E值 = CSV编码值 + 对应通道的拼孔值
                        original_e = int(row['encoding']) + offset
                        
                        points.append({
                            "e": original_e,
                            "channel": channel,
                            "depth": float(depth),
                            "amplitude": 100.0  # 默认幅度值
                        })
                    
                    # 生成伤损结构
                    if points:
                        # 确定伤损类型
                        if main_label == 0:
                            annotation_type = "结构"
                        else:
                            annotation_type = "伤损"
                        
                        annotation = {
                            "id": str(cluster_id),
                            "type": annotation_type,
                            "subtype": {
                                "id": str(main_label),
                                "name": self.flaw_types.get(main_label, "未知类型")
                            },
                            "points": points,
                            "start_E": min(p['e'] for p in points),
                            "end_E": max(p['e'] for p in points),
                            "channels": [channel],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        annotations.append(annotation)
                        cluster_id += 1
            else:
                # 单通道标签处理（向后兼容）
                # 确定簇的主要类型（出现次数最多的标签）
                label_col = 'predicted_label' if 'predicted_label' in clustered_df.columns else 'label'
                main_label = cluster_points[label_col].mode().iloc[0] if not cluster_points.empty else 0
                
                # 转换为伤损结构
                points = []
                for _, row in cluster_points.iterrows():
                    # 找到第一个非零的通道深度值
                    channel = None
                    depth = None
                    for i in range(9):
                        if row[f'channel_{i}'] != 0:
                            channel = i  # 通道号直接使用0-8
                            depth = row[f'channel_{i}']
                            break
                    
                    if channel is not None and depth is not None:
                        # 获取该通道的拼孔值
                        offset = beam_offset_map.get(channel, 0)
                        # 逆拼孔处理：原始E值 = CSV编码值 + 对应通道的拼孔值
                        original_e = int(row['encoding']) + offset
                        
                        points.append({
                            "e": original_e,
                            "channel": channel,
                            "depth": float(depth),
                            "amplitude": 100.0  # 默认幅度值
                        })
                
                # 生成伤损结构
                if points:
                    # 确定伤损类型
                    if main_label == 0:
                        annotation_type = "结构"
                    else:
                        annotation_type = "伤损"
                    
                    annotation = {
                        "id": str(cluster_id),
                        "type": annotation_type,
                        "subtype": {
                            "id": str(main_label),
                            "name": self.flaw_types.get(main_label, "未知类型")
                        },
                        "points": points,
                        "start_E": min(p['e'] for p in points),
                        "end_E": max(p['e'] for p in points),
                        "channels": list(set(p['channel'] for p in points)),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    annotations.append(annotation)
                    cluster_id += 1
        
        # 合并重复或重叠的标注
        merged_annotations = self.merge_overlapping_annotations(annotations)
        print(f"合并后生成 {len(merged_annotations)} 个伤损结构")
        
        # 生成最终的hdc.json结构
        hdc_json = {
            "annotations": merged_annotations
        }
        
        return hdc_json
    
    def merge_overlapping_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """
        合并相邻或重叠的伤损结构，减少重复标注
        
        Args:
            annotations: 原始标注列表
            
        Returns:
            合并后的标注列表
        """
        if not annotations:
            return annotations
        
        # 按通道和start_E排序标注
        sorted_annotations = sorted(annotations, key=lambda x: (x['channels'][0], x['start_E']))
        
        merged = []
        current = sorted_annotations[0].copy()
        
        for next_annotation in sorted_annotations[1:]:
            # 检查当前标注和下一个标注是否可以合并
            # 条件1：通道相同或相关
            channels_overlap = set(current['channels']).intersection(set(next_annotation['channels']))
            # 条件2：E值范围重叠或相邻（E值差不超过20）
            e_overlap = not (current['end_E'] < next_annotation['start_E'] - 20 or next_annotation['end_E'] < current['start_E'] - 20)
            # 条件3：伤损类型相同
            type_same = current['subtype']['id'] == next_annotation['subtype']['id']
            
            if (channels_overlap or abs(current['channels'][0] - next_annotation['channels'][0]) <= 1) and e_overlap and type_same:
                # 合并标注
                current['start_E'] = min(current['start_E'], next_annotation['start_E'])
                current['end_E'] = max(current['end_E'], next_annotation['end_E'])
                current['points'].extend(next_annotation['points'])
                current['channels'] = list(set(current['channels']).union(set(next_annotation['channels'])))
                current['channels'].sort()
            else:
                # 无法合并，保存当前标注，开始处理下一个
                merged.append(current)
                current = next_annotation.copy()
        
        # 保存最后一个标注
        merged.append(current)
        
        # 进一步优化：合并相同类型、相同通道且E值连续的标注
        final_merged = []
        if merged:
            final_merged.append(merged[0])
            for annotation in merged[1:]:
                prev_annotation = final_merged[-1]
                
                # 检查是否可以进一步合并
                channels_same = set(prev_annotation['channels']) == set(annotation['channels'])
                type_same = prev_annotation['subtype']['id'] == annotation['subtype']['id']
                e_continuous = annotation['start_E'] - prev_annotation['end_E'] <= 10
                
                if channels_same and type_same and e_continuous:
                    # 合并连续的标注
                    prev_annotation['end_E'] = annotation['end_E']
                    prev_annotation['points'].extend(annotation['points'])
                else:
                    final_merged.append(annotation)
        
        return final_merged
    
    def generate_hdc_json_direct(self, defect_points: pd.DataFrame, original_csv_path: str) -> Dict:
        """
        直接生成hdc.json格式的数据，跳过聚类步骤
        
        Args:
            defect_points: 缺陷点DataFrame
            original_csv_path: 原始CSV文件路径
            
        Returns:
            hdc.json格式的字典
        """
        annotations = []
        annotation_id = 1000  # 自动标注ID从1000开始
        
        # 检查是否有多通道标签
        has_multichannel_pred = all([f'predicted_label_{i}' in defect_points.columns for i in range(9)])
        has_multichannel_true = all([f'label_{i}' in defect_points.columns for i in range(9)])
        
        # 从原始JSON文件中提取超声波参数
        csv_path = Path(original_csv_path)
        original_stem = csv_path.stem.replace('_hdc', '')
        original_json_path = str(csv_path.parent / f"{original_stem}.json")
        beam_offset_map, beam_way_map, beam_angle_map = self.extract_beam_parameters(original_json_path)
        print(f"Using beam offset map: {beam_offset_map} for {original_csv_path}")
        
        # 直接处理每个缺陷点
        for _, row in defect_points.iterrows():
            # 获取当前行的编码值
            encoding = int(row['encoding'])
            
            if has_multichannel_pred or has_multichannel_true:
                # 多通道标签处理
                for channel in range(9):
                    # 获取该通道的深度值
                    if row[f'channel_{channel}'] == 0:
                        continue
                    
                    depth = row[f'channel_{channel}']
                    
                    # 获取该通道的标签
                    if has_multichannel_pred:
                        label = row[f'predicted_label_{channel}']
                    else:
                        label = row[f'label_{channel}']
                    
                    # 如果标签为0，跳过（正常点不生成标注）
                    if label == 0:
                        continue
                    
                    # 获取该通道的拼孔值
                    offset = beam_offset_map.get(channel, 0)
                    # 逆拼孔处理：原始E值 = CSV编码值 + 对应通道的拼孔值
                    original_e = encoding + offset
                    
                    # 生成伤损结构
                    points = [{
                        "e": original_e,
                        "channel": channel,
                        "depth": float(depth),
                        "amplitude": 100.0  # 默认幅度值
                    }]
                    
                    # 确定伤损类型
                    if label == 0:
                        annotation_type = "结构"
                    else:
                        annotation_type = "伤损"
                    
                    annotation = {
                        "id": str(annotation_id),
                        "type": annotation_type,
                        "subtype": {
                            "id": str(label),
                            "name": self.flaw_types.get(label, "未知类型")
                        },
                        "points": points,
                        "start_E": original_e,
                        "end_E": original_e,
                        "channels": [channel],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    annotations.append(annotation)
                    annotation_id += 1
            else:
                # 单通道标签处理（向后兼容）
                # 找到非零深度值的通道和深度
                channel = None
                depth = None
                for i in range(9):
                    if row[f'channel_{i}'] != 0:
                        channel = i
                        depth = row[f'channel_{i}']
                        break
                
                if channel is not None and depth is not None:
                    # 确定使用的标签列
                    if has_multichannel_pred:
                        label = row[f'predicted_label_{channel}']
                    elif has_multichannel_true:
                        label = row[f'label_{channel}']
                    else:
                        label_col = 'predicted_label' if 'predicted_label' in row.index else 'label'
                        label = row[label_col]
                    
                    # 获取该通道的拼孔值
                    offset = beam_offset_map.get(channel, 0)
                    # 逆拼孔处理：原始E值 = CSV编码值 + 对应通道的拼孔值
                    original_e = encoding + offset
                    
                    # 生成伤损结构
                    points = [{
                        "e": original_e,
                        "channel": channel,
                        "depth": float(depth),
                        "amplitude": 100.0  # 默认幅度值
                    }]
                    
                    # 确定伤损类型
                    if label == 0:
                        annotation_type = "结构"
                    else:
                        annotation_type = "伤损"
                    
                    annotation = {
                        "id": str(annotation_id),
                        "type": annotation_type,
                        "subtype": {
                            "id": str(label),
                            "name": self.flaw_types.get(label, "未知类型")
                        },
                        "points": points,
                        "start_E": original_e,
                        "end_E": original_e,
                        "channels": [channel],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    annotations.append(annotation)
                    annotation_id += 1
        
        # 生成最终的hdc.json结构
        hdc_json = {
            "annotations": annotations
        }
        
        return hdc_json
    
    def convert_csv_to_hdc_json(self, csv_file: str, output_file: str = None, eps: float = 5.0, min_samples: int = 5):
        """
        将CSV文件转换为hdc.json文件
        
        Args:
            csv_file: 输入CSV文件路径
            output_file: 输出JSON文件路径，默认与CSV文件同目录
            eps: DBSCAN的ε参数（已屏蔽聚类，此参数无效）
            min_samples: DBSCAN的min_samples参数（已屏蔽聚类，此参数无效）
            
        Returns:
            输出JSON文件路径
        """
        # 设置默认输出文件名，确保只添加一个hdc后缀
        if output_file is None:
            csv_path = Path(csv_file)
            output_file = str(csv_path.parent / f"{csv_path.stem.replace('_hdc', '')}_hdc.json")
        
        print(f"正在处理文件: {csv_file}")
        
        # 1. 加载CSV文件
        df = self.load_csv(csv_file)
        
        # 2. 提取缺陷点
        defect_points = self.extract_defect_points(df)
        print(f"找到 {len(defect_points)} 个缺陷点")
        
        # 3. 根据需要选择是否聚类
        # 默认进行聚类，解决多个窗口检测同一目标多次的问题
        skip_clustering = False
        
        if skip_clustering:
            # 跳过聚类，直接生成hdc.json
            hdc_json = self.generate_hdc_json_direct(defect_points, csv_file)
        else:
            # 从原始JSON文件中提取超声波参数
            csv_path = Path(csv_file)
            original_stem = csv_path.stem.replace('_hdc', '')
            original_json_path = str(csv_path.parent / f"{original_stem}.json")
            beam_offset_map, beam_way_map, beam_angle_map = self.extract_beam_parameters(original_json_path)
            
            # 进行聚类，调整DBSCAN参数优化聚类效果
            clustered_df = self.cluster_points(defect_points, eps=15.0, min_samples=2,
                                             beam_offset_map=beam_offset_map,
                                             beam_way_map=beam_way_map,
                                             beam_angle_map=beam_angle_map)
            print(f"聚类完成，生成 {len(clustered_df['cluster'].unique()) - 1} 个簇")
            
            # 生成hdc.json
            hdc_json = self.generate_hdc_json(clustered_df, csv_file)
        print(f"生成 {len(hdc_json['annotations'])} 个伤损结构")
        
        # 4. 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hdc_json, f, ensure_ascii=False, indent=2)
        
        print(f"hdc.json文件已保存到: {output_file}")
        return output_file
    
    def batch_convert(self, input_dir: str, output_dir: str = None, eps: float = 5.0, min_samples: int = 5):
        """
        批量转换目录中的所有CSV文件
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，默认与输入目录相同
            eps: DBSCAN的ε参数
            min_samples: DBSCAN的min_samples参数
            
        Returns:
            转换后的文件列表
        """
        input_path = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 获取所有CSV文件
        csv_files = list(input_path.glob("*.csv"))
        
        converted_files = []
        for csv_file in csv_files:
            # 跳过标注文件
            if '标注' in csv_file.name:
                continue
            
            # 转换文件，确保只添加一个hdc后缀
            output_file = str(output_path / f"{csv_file.stem.replace('_hdc', '')}_hdc.json")
            converted_file = self.convert_csv_to_hdc_json(
                str(csv_file), output_file, eps=eps, min_samples=min_samples
            )
            converted_files.append(converted_file)
        
        print(f"\n批量转换完成，共处理 {len(converted_files)} 个文件")
        return converted_files


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将CSV格式的超声波HDC检测结果转换为hdc.json格式')
    parser.add_argument('--csv_file', help='单个CSV文件路径')
    parser.add_argument('--input_dir', help='输入目录路径（批量转换）')
    parser.add_argument('--output_file', help='输出JSON文件路径')
    parser.add_argument('--output_dir', help='输出目录路径（批量转换）')
    parser.add_argument('--eps', type=float, default=5.0, help='DBSCAN的ε参数，默认5.0')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN的min_samples参数，默认5')
    
    args = parser.parse_args()
    
    converter = CsvToHdcJsonConverter()
    
    if args.csv_file:
        # 转换单个文件
        converter.convert_csv_to_hdc_json(
            args.csv_file, args.output_file, eps=args.eps, min_samples=args.min_samples
        )
    elif args.input_dir:
        # 批量转换
        converter.batch_convert(
            args.input_dir, args.output_dir, eps=args.eps, min_samples=args.min_samples
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()