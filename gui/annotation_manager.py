#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标注管理类
用于处理标注数据的生成、保存和加载
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any


class AnnotationManager:
    """标注管理类"""
    
    def __init__(self):
        """初始化标注管理器"""
        # 标注数据列表
        self.annotations = []
        
        # 标注ID计数器
        self.annotation_id = 1
    
    def create_annotation(self, annotation_type: str, subtype: Dict, points: List[Dict]):
        """
        创建标注数据
        
        Args:
            annotation_type: 标注类型（"伤损"或"结构"）
            subtype: 子类型信息
            points: 选中的点列表
        
        Returns:
            Dict: 标注数据
        """
        if not points:
            return None
        
        # 按编码值排序
        sorted_points = sorted(points, key=lambda p: p['e'])
        
        # 获取编码值范围
        start_e = sorted_points[0]['e']
        end_e = sorted_points[-1]['e']
        
        # 获取涉及的通道
        channels = sorted(list(set(p['channel'] for p in sorted_points)))
        
        # 创建标注数据
        annotation = {
            "id": str(self.annotation_id),
            "type": annotation_type,
            "subtype": subtype,
            "points": sorted_points,
            "start_E": start_e,
            "end_E": end_e,
            "channels": channels,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加到标注列表
        self.annotations.append(annotation)
        
        # 增加ID计数器
        self.annotation_id += 1
        
        return annotation
    
    def save_annotations(self, file_path: str):
        """
        保存标注数据到JSON文件
        
        Args:
            file_path: 输出文件路径
        
        Returns:
            bool: 保存成功返回True，失败返回False
        """
        try:
            # 直接保存当前内存中的所有标注，包括删除操作后的结果
            # 不合并现有文件，确保删除的标注不会出现在保存的文件中
            all_annotations = self.annotations
            
            # 保存所有标注
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "annotations": all_annotations,
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0"
                }, f, ensure_ascii=False, indent=2)
            
            # 保存成功后不清空标注列表，保留所有标注
            # 但更新ID计数器，确保新标注的ID唯一
            if all_annotations:
                max_id = max(int(anno["id"]) for anno in all_annotations)
                self.annotation_id = max_id + 1
            
            return True
        except Exception as e:
            print(f"保存标注数据失败: {str(e)}")
            return False
    
    def load_annotations(self, file_path: str, replace: bool = True):
        """
        从JSON文件加载标注数据
        
        Args:
            file_path: 输入文件路径
            replace: 是否替换现有标注，False则合并到现有标注中
        
        Returns:
            bool: 加载成功返回True，失败返回False
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式
            if "annotations" in data:
                loaded_annotations = data["annotations"]
                
                if replace:
                    # 替换现有标注
                    self.annotations = loaded_annotations
                else:
                    # 合并到现有标注中，确保标注ID唯一性
                    # 先收集所有已存在的ID
                    existing_ids = set(anno["id"] for anno in self.annotations)
                    
                    # 计算当前最大ID，用于生成新ID
                    current_max_id = max(int(anno["id"]) for anno in self.annotations) if self.annotations else 0
                    next_id = current_max_id + 1
                    
                    # 处理每个加载的标注
                    for anno in loaded_annotations:
                        if anno["id"] not in existing_ids:
                            # ID不存在，直接添加
                            self.annotations.append(anno)
                            existing_ids.add(anno["id"])
                            # 更新最大ID
                            if int(anno["id"]) > current_max_id:
                                current_max_id = int(anno["id"])
                                next_id = current_max_id + 1
                        else:
                            # ID已存在，生成新的唯一ID
                            new_id = str(next_id)
                            while new_id in existing_ids:
                                next_id += 1
                                new_id = str(next_id)
                            
                            # 更新标注的ID
                            anno_copy = anno.copy()
                            anno_copy["id"] = new_id
                            
                            # 添加到标注列表
                            self.annotations.append(anno_copy)
                            existing_ids.add(new_id)
                            
                            # 更新ID计数器
                            next_id += 1
                            current_max_id = next_id - 1
                
                # 更新ID计数器
                if self.annotations:
                    max_id = max(int(anno["id"]) for anno in self.annotations)
                    self.annotation_id = max_id + 1
                else:
                    self.annotation_id = 1
                
                return True
            else:
                print("无效的标注数据格式")
                return False
        except Exception as e:
            print(f"加载标注数据失败: {str(e)}")
            return False
    
    def delete_annotation(self, annotation_id: str):
        """
        删除标注数据
        
        Args:
            annotation_id: 标注ID
        
        Returns:
            bool: 删除成功返回True，失败返回False
        """
        for i, annotation in enumerate(self.annotations):
            if annotation["id"] == annotation_id:
                del self.annotations[i]
                return True
        return False
    
    def get_annotations(self):
        """
        获取所有标注数据
        
        Returns:
            List[Dict]: 标注数据列表
        """
        return self.annotations
    
    def get_annotation_by_id(self, annotation_id: str):
        """
        根据ID获取标注数据
        
        Args:
            annotation_id: 标注ID
        
        Returns:
            Dict: 标注数据，如果不存在返回None
        """
        for annotation in self.annotations:
            if annotation["id"] == annotation_id:
                return annotation
        return None
    
    def clear_annotations(self):
        """
        清空所有标注数据
        """
        self.annotations.clear()
        self.annotation_id = 1
    
    def get_annotations_in_range(self, start_e: int, end_e: int):
        """
        获取指定编码值范围内的标注数据
        
        Args:
            start_e: 起始编码值
            end_e: 结束编码值
        
        Returns:
            List[Dict]: 标注数据列表
        """
        return [
            annotation for annotation in self.annotations
            if annotation["start_E"] <= end_e and annotation["end_E"] >= start_e
        ]
    
    def export_annotations_as_text(self):
        """
        导出标注数据为文本格式
        
        Returns:
            str: 文本格式的标注数据
        """
        result = []
        
        for annotation in self.annotations:
            result.append(f"=== 标注ID: {annotation['id']} ===")
            result.append(f"类型: {annotation['type']}")
            result.append(f"子类型: {annotation['subtype']['name']}")
            result.append(f"编码范围: E{annotation['start_E']} - E{annotation['end_E']}")
            result.append(f"通道: {annotation['channels']}")
            result.append(f"点数: {len(annotation['points'])}")
            result.append(f"时间: {annotation['timestamp']}")
            result.append("")
            
            # 按编码值分组
            e_groups = {}
            for p in annotation['points']:
                e = p['e']
                if e not in e_groups:
                    e_groups[e] = []
                e_groups[e].append(p)
            
            # 生成文本格式
            for e in sorted(e_groups.keys()):
                points = e_groups[e]
                for p in points:
                    b_type = "B1" if p.get('b_type', 'B1') == 'B1' else "B2"
                    result.append(f"E{e} {b_type}")
            
            result.append("")
        
        return "\n".join(result)