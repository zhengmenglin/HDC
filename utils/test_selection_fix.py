#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的点选择功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from point_selector import PointSelector

class MockCanvas:
    """模拟画布类"""
    def __init__(self):
        self.displayed_points = [
            {'original': {'e': 100, 'channel': 0, 'depth': 50.0, 'amplitude': 80.0}, 'e': 100.0, 'depth': 50.0, 'amplitude': 80.0, 'b_type': 'B1'},
            {'original': {'e': 101, 'channel': 1, 'depth': 55.0, 'amplitude': 85.0}, 'e': 101.0, 'depth': 55.0, 'amplitude': 85.0, 'b_type': 'B1'},
            {'original': {'e': 102, 'channel': 0, 'depth': 60.0, 'amplitude': 90.0}, 'e': 102.0, 'depth': 60.0, 'amplitude': 90.0, 'b_type': 'B1'},
        ]
        self.selected_points = []

    def draw(self):
        pass

    def mpl_connect(self, event, handler):
        pass

def test_callback():
    """测试回调函数"""
    print("回调函数被调用")

def test_selection_logic():
    """测试选择逻辑"""
    print("测试修复后的点选择逻辑...")

    # 创建模拟画布
    canvas = MockCanvas()

    # 创建点选择器（跳过事件连接）
    selector = PointSelector.__new__(PointSelector)
    selector.canvas = canvas
    selector.axes = None
    selector.selected_points = []
    selector.is_selecting = False
    selector.is_multi_select = False
    selector.callback = test_callback

    # 测试1：选择第一个点
    print("\n1. 选择第一个点 (100, 50)")
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")
    if selector.selected_points:
        print(f"选中点: E={selector.selected_points[0]['original']['e']}, Ch={selector.selected_points[0]['original']['channel']}")

    # 测试2：再次选择同一个点（应该取消选择）
    print("\n2. 再次选择第一个点 (100, 50) - 应该取消选择")
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")

    # 测试3：选择第二个点
    print("\n3. 选择第二个点 (101, 55)")
    selector.select_single_point(101, 55)
    print(f"选中点数量: {len(selector.selected_points)}")
    if selector.selected_points:
        print(f"选中点: E={selector.selected_points[0]['original']['e']}, Ch={selector.selected_points[0]['original']['channel']}")

    # 测试4：进入多选模式，选择第一个点
    print("\n4. 进入多选模式，选择第一个点")
    selector.is_multi_select = True
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")
    for i, p in enumerate(selector.selected_points):
        print(f"  点{i}: E={p['original']['e']}, Ch={p['original']['channel']}")

    # 测试5：在多选模式下再次选择第一个点（应该取消选择）
    print("\n5. 在多选模式下再次选择第一个点 - 应该取消选择")
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")
    for i, p in enumerate(selector.selected_points):
        print(f"  点{i}: E={p['original']['e']}, Ch={p['original']['channel']}")

    print("\n测试完成!")

if __name__ == "__main__":
    test_selection_logic()