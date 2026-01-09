#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试点选择功能
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
        self.axes = None

    def draw(self):
        print(f"画布重绘，选中点数量: {len(self.selected_points)}")

    def mpl_connect(self, event, handler):
        # 模拟matplotlib的连接方法
        pass

def test_callback():
    """测试回调函数"""
    print("回调函数被调用")

def test_point_selection():
    """测试点选择功能"""
    print("开始测试点选择功能...")

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

    # 测试选择第一个点
    print("\n1. 选择第一个点 (100, 50)")
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")

    # 测试再次选择第一个点（应该取消选择）
    print("\n2. 再次选择第一个点 (100, 50) - 应该取消选择")
    print(f"选择前选中点数量: {len(selector.selected_points)}")
    for i, p in enumerate(selector.selected_points):
        print(f"  选中点{i}: E={p['original']['e']}, Ch={p['original']['channel']}")

    closest = selector.find_closest_point(100, 50)
    if closest:
        print(f"找到点: E={closest['original']['e']}, Ch={closest['original']['channel']}")
        # 检查是否与已选点相等
        for selected in selector.selected_points:
            equal = selector.points_equal(closest, selected)
            print(f"与选中点比较: {equal}")

    # 手动执行选择逻辑来调试
    closest_point = selector.find_closest_point(100, 50)
    if closest_point:
        print(f"执行选择逻辑...")
        print(f"  closest_point: {closest_point['original']}")

        # 检查点是否已经被选中
        point_already_selected = False
        for selected_point in selector.selected_points:
            if selector.points_equal(closest_point, selected_point):
                point_already_selected = True
                print(f"  点已被选中，将取消选择")
                break

        print(f"  point_already_selected: {point_already_selected}")

        if point_already_selected:
            old_count = len(selector.selected_points)
            selector.selected_points = [p for p in selector.selected_points
                                       if not selector.points_equal(closest_point, p)]
            new_count = len(selector.selected_points)
            print(f"  取消选择: {old_count} -> {new_count}")

            # 手动调用高亮显示
            selector.highlight_selected_points()
            if selector.callback:
                selector.callback()
        else:
            selector.selected_points.append(closest_point)
            selector.highlight_selected_points()
            if selector.callback:
                selector.callback()

    print(f"选择后选中点数量: {len(selector.selected_points)}")
    for i, p in enumerate(selector.selected_points):
        print(f"  选中点{i}: E={p['original']['e']}, Ch={p['original']['channel']}")

    # 测试选择第二个点
    print("\n3. 选择第二个点 (101, 55)")
    selector.select_single_point(101, 55)
    print(f"选中点数量: {len(selector.selected_points)}")

    # 测试选择第三个点
    print("\n4. 选择第三个点 (102, 60)")
    selector.select_single_point(102, 60)
    print(f"选中点数量: {len(selector.selected_points)}")

    # 测试多选模式
    print("\n5. 进入多选模式，选择第一个点")
    selector.is_multi_select = True
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")

    # 再次选择第一个点取消选择
    print("\n6. 在多选模式下再次选择第一个点 - 取消选择")
    selector.select_single_point(100, 50)
    print(f"选中点数量: {len(selector.selected_points)}")

    print("\n测试完成!")

if __name__ == "__main__":
    test_point_selection()