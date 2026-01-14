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
        
        Args:
            df: 包含检测结果的DataFrame
            
        Returns:
            缺陷点的DataFrame
        """
        # 检查是否有predicted_label列，优先使用预测结果
        if 'predicted_label' in df.columns:
            return df[df['predicted_label'] != 0].copy()
        else:
            return df[df['label'] != 0].copy()
    
    def cluster_points(self, df: pd.DataFrame, eps: float = 5.0, min_samples: int = 5) -> pd.DataFrame:
        """
        使用DBSCAN聚类算法将同一目标的点归为一个簇
        
        Args:
            df: 缺陷点DataFrame
            eps: DBSCAN的ε参数，两个样本被视为同一簇的最大距离
            min_samples: DBSCAN的min_samples参数，一个簇的最小样本数
            
        Returns:
            带有簇标签的DataFrame
        """
        if df.empty:
            df['cluster'] = []
            return df
        
        # 准备聚类特征：编码值和通道深度值
        # 选择编码值和所有通道的深度值作为特征
        feature_cols = ['encoding'] + [f'channel_{i}' for i in range(9)]
        X = df[feature_cols].values
        
        # 使用DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        
        # 将簇标签添加到DataFrame
        df['cluster'] = clusters
        
        return df
    
    def extract_beam_offset(self, original_json_path: str) -> Dict[int, float]:
        """
        从原始JSON文件中提取拼孔值（BeamOffset）
        
        Args:
            original_json_path: 原始JSON文件路径
            
        Returns:
            通道到拼孔值的映射
        """
        beam_offset_map = {}
        
        if not os.path.exists(original_json_path):
            return beam_offset_map
        
        try:
            with open(original_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取配置信息（第一个元素通常是配置）
            if data and isinstance(data[0], dict):
                config_item = data[0]
                if 'E' in config_item and isinstance(config_item['E'], int) and config_item['E'] == 1:
                    if 'BeamOffset' in config_item and isinstance(config_item['BeamOffset'], list):
                        beam_offset = config_item['BeamOffset']
                        # BeamOffset格式：[通道号, 拼孔值, 通道号, 拼孔值, ...]
                        for i in range(0, len(beam_offset), 2):
                            if i + 1 < len(beam_offset):
                                channel = int(beam_offset[i])
                                offset = beam_offset[i + 1]
                                beam_offset_map[channel] = offset
        except Exception as e:
            print(f"提取拼孔值时出错: {e}")
        
        return beam_offset_map
    
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
        
        # 检查是否有predicted_label列，优先使用预测结果
        label_col = 'predicted_label' if 'predicted_label' in clustered_df.columns else 'label'
        
        # 从原始JSON文件中提取拼孔值
        # 构造原始JSON文件路径：将_csv_hdc.csv替换为原始扩展名
        csv_path = Path(original_csv_path)
        # 处理文件名：例如 1_hdc.csv -> 1.json
        original_stem = csv_path.stem.replace('_hdc', '')
        original_json_path = str(csv_path.parent / f"{original_stem}.json")
        # 提取拼孔值映射
        beam_offset_map = self.extract_beam_offset(original_json_path)
        print(f"Using beam offset map: {beam_offset_map} for {original_csv_path}")
        
        # 按簇分组
        for cluster in clustered_df['cluster'].unique():
            if cluster == -1:  # 噪声点，跳过
                continue
                
            cluster_points = clustered_df[clustered_df['cluster'] == cluster]
            if cluster_points.empty:
                continue
                
            # 确定簇的主要类型（出现次数最多的标签）
            main_label = cluster_points[label_col].mode().iloc[0] if not cluster_points.empty else 0
            
            # 转换为伤损结构
            points = []
            for _, row in cluster_points.iterrows():
                # 找到第一个非零的通道深度值
                channel = None
                depth = None
                for i in range(9):
                    if row[f'channel_{i}'] != 0:
                        channel = i + 1  # 通道号从1开始
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
                
                # 提取文件名作为参考
                base_name = os.path.basename(original_csv_path)
                
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
            eps: DBSCAN的ε参数
            min_samples: DBSCAN的min_samples参数
            
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
        
        # 3. 使用DBSCAN聚类
        clustered_df = self.cluster_points(defect_points, eps=eps, min_samples=min_samples)
        
        # 4. 生成hdc.json
        hdc_json = self.generate_hdc_json(clustered_df, csv_file)
        print(f"生成 {len(hdc_json['annotations'])} 个伤损结构")
        
        # 5. 保存为JSON文件
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