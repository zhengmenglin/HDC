#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试AnnotationManager的标注合并功能
验证ID冲突处理是否正常工作
"""

import os
import tempfile
import json
from gui.annotation_manager import AnnotationManager

# 创建测试用的标注数据
def create_test_annotations():
    # 常规标注（模拟标注.json）
    regular_annotations = {
        "annotations": [
            {
                "id": "1",
                "type": "结构",
                "subtype": {"id": "101", "name": "螺孔"},
                "points": [{"e": 100, "channel": 1, "depth": 50.0, "amplitude": 50.0}],
                "start_E": 100,
                "end_E": 100,
                "channels": [1],
                "timestamp": "2026-01-14T10:00:00"
            },
            {
                "id": "2",
                "type": "结构",
                "subtype": {"id": "101", "name": "螺孔"},
                "points": [{"e": 200, "channel": 2, "depth": 60.0, "amplitude": 60.0}],
                "start_E": 200,
                "end_E": 200,
                "channels": [2],
                "timestamp": "2026-01-14T10:00:01"
            }
        ]
    }
    
    # HDC标注（模拟_hdc.json），包含ID冲突
    hdc_annotations = {
        "annotations": [
            {
                "id": "1",  # 与常规标注ID冲突
                "type": "伤损",
                "subtype": {"id": "13", "name": "螺孔"},
                "points": [{"e": 150, "channel": 1, "depth": 55.0, "amplitude": 100.0}],
                "start_E": 150,
                "end_E": 150,
                "channels": [1],
                "timestamp": "2026-01-14T13:00:00"
            },
            {
                "id": "3",  # 新ID，无冲突
                "type": "伤损",
                "subtype": {"id": "13", "name": "螺孔"},
                "points": [{"e": 250, "channel": 2, "depth": 65.0, "amplitude": 100.0}],
                "start_E": 250,
                "end_E": 250,
                "channels": [2],
                "timestamp": "2026-01-14T13:00:01"
            }
        ]
    }
    
    return regular_annotations, hdc_annotations

def test_annotation_merge():
    """测试标注合并功能"""
    print("测试标注合并功能开始...")
    
    # 创建AnnotationManager实例
    annotation_manager = AnnotationManager()
    
    # 创建临时文件
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试标注文件
        regular_file = os.path.join(temp_dir, "test_标注.json")
        hdc_file = os.path.join(temp_dir, "test_hdc.json")
        
        # 生成测试数据
        regular_annotations, hdc_annotations = create_test_annotations()
        
        # 写入测试文件
        with open(regular_file, 'w', encoding='utf-8') as f:
            json.dump(regular_annotations, f, ensure_ascii=False, indent=2)
        
        with open(hdc_file, 'w', encoding='utf-8') as f:
            json.dump(hdc_annotations, f, ensure_ascii=False, indent=2)
        
        print(f"创建测试文件: {regular_file}")
        print(f"创建测试文件: {hdc_file}")
        
        # 加载常规标注（replace=True，默认）
        annotation_manager.load_annotations(regular_file)
        print(f"加载常规标注后，标注数量: {len(annotation_manager.get_annotations())}")
        
        # 加载HDC标注（replace=False）
        annotation_manager.load_annotations(hdc_file, replace=False)
        print(f"加载HDC标注后，标注数量: {len(annotation_manager.get_annotations())}")
        
        # 检查所有标注ID是否唯一
        all_annotations = annotation_manager.get_annotations()
        all_ids = [anno['id'] for anno in all_annotations]
        unique_ids = set(all_ids)
        
        print(f"所有标注ID: {all_ids}")
        print(f"唯一ID数量: {len(unique_ids)}")
        
        # 验证结果
        if len(all_annotations) == 4 and len(unique_ids) == 4:
            print("✅ 测试通过：标注成功合并，所有ID唯一")
            return True
        else:
            print("❌ 测试失败：标注合并失败或存在ID冲突")
            return False

if __name__ == "__main__":
    test_annotation_merge()
