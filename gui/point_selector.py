#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点选择功能
用于在B显图像上选择和高亮显示点
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector


class PointSelector:
    """点选择类"""
    
    def __init__(self, canvas, callback=None):
        """
        初始化点选择器

        Args:
            canvas: BScanCanvas实例
            callback: 选择变化时的回调函数
        """
        self.canvas = canvas
        self.axes = canvas.axes

        # 选中的点
        self.selected_points = []

        # 点的选择状态
        self.is_selecting = False
        self.is_multi_select = True  # 默认启用多选模式

        # 矩形选择器相关变量
        self.rect_selector = None
        self.is_dragging = False
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        # 框选功能开关，默认开启
        self.rect_select_enabled = True
        # 记录鼠标按键类型（1: 左键，3: 右键）
        self.mouse_button = None

        # 选择变化回调函数
        self.callback = callback

        # 连接鼠标事件
        self.connect_events()
        
    def set_rect_select_enabled(self, enabled):
        """
        设置框选功能状态
        
        Args:
            enabled: bool, True表示开启框选功能，False表示关闭框选功能
        """
        self.rect_select_enabled = enabled
    
    def connect_events(self):
        """连接鼠标事件"""
        # 鼠标点击事件
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion_notify)
        
        # 键盘事件
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
    
    def on_button_press(self, event):
        """鼠标按下事件处理"""
        # 只处理左键和右键点击
        if event.button not in [1, 3]:  # 左键是1，右键是3
            return
        
        # 检查是否在坐标轴范围内
        if not event.inaxes == self.axes:
            return
        
        # 记录点击位置和鼠标按键类型
        self.start_x = event.xdata
        self.start_y = event.ydata
        self.end_x = event.xdata
        self.end_y = event.ydata
        self.mouse_button = event.button
        self.is_dragging = True
    
    def on_button_release(self, event):
        """鼠标释放事件处理"""
        # 只处理左键和右键点击
        if event.button not in [1, 3]:  # 左键是1，右键是3
            return
        
        # 检查是否在坐标轴范围内
        if not event.inaxes == self.axes:
            return
        
        # 记录释放位置
        self.end_x = event.xdata
        self.end_y = event.ydata
        
        # 计算点击位置和释放位置的距离
        distance = np.sqrt((self.end_x - self.start_x) ** 2 + (self.end_y - self.start_y) ** 2)
        
        # 如果距离很小，认为是单击
        if distance < 5:  # 阈值可以调整
            self.select_single_point(self.start_x, self.start_y)
        else:
            # 否则认为是拖拽，根据开关状态决定是否进行框选
            if self.rect_select_enabled:
                # 根据鼠标按键类型决定操作
                # 左键框选：添加点
                # 右键框选：移除点
                add_points = (event.button == 1)
                self.select_points_by_rect(self.start_x, self.start_y, self.end_x, self.end_y, add_points)
            else:
                # 框选功能关闭，不执行框选
                pass
        
        self.is_dragging = False
    
    def on_key_press(self, event):
        """键盘按下事件处理"""
        if event.key == 'escape':
            # 取消选择
            self.clear_selection()
    
    def on_key_release(self, event):
        """键盘释放事件处理"""
        # 移除所有键盘事件处理逻辑，只保留对escape键的处理
        pass
    
    def on_motion_notify(self, event):
        """鼠标拖动事件处理"""
        if not self.is_dragging:
            return
        
        # 检查是否在坐标轴范围内
        if not event.inaxes == self.axes:
            return
        
        # 更新释放位置
        self.end_x = event.xdata
        self.end_y = event.ydata
        
        # 只有当框选功能开启时，才绘制框选区域
        if self.rect_select_enabled:
            self.draw_selection_rectangle()
    
    def select_single_point(self, x, y):
        """选择单个点"""
        # 找到最近的点
        closest_point = self.find_closest_point(x, y)

        if closest_point:
            # 检查点是否已经被选中（基于内容比较）
            point_already_selected = False
            for selected_point in self.selected_points:
                if self.points_equal(closest_point, selected_point):
                    point_already_selected = True
                    break

            if point_already_selected:
                # 如果点已经被选中，取消选择（移除所有匹配的点）
                self.selected_points = [p for p in self.selected_points
                                       if not self.points_equal(closest_point, p)]
            else:
                # 如果点没有被选中
                # 如果不是多选模式，先清空选择
                if not self.is_multi_select:
                    self.selected_points.clear()

                # 然后添加到选择列表
                self.selected_points.append(closest_point)

            # 高亮显示选中的点
            self.highlight_selected_points()

            # 调用回调函数
            if self.callback:
                self.callback()

    def points_equal(self, point1, point2):
        """比较两个点是否相等（基于原始数据）"""
        if not point1 or not point2:
            return False

        # 比较原始数据的关键字段
        orig1 = point1.get('original', {})
        orig2 = point2.get('original', {})

        return (orig1.get('e') == orig2.get('e') and
                orig1.get('channel') == orig2.get('channel') and
                abs(orig1.get('depth', 0) - orig2.get('depth', 0)) < 0.01 and
                abs(orig1.get('amplitude', 0) - orig2.get('amplitude', 0)) < 0.01)

    def draw_selection_rectangle(self):
        """绘制框选区域"""
        # 先更新B显图像，清除之前的框选区域
        self.canvas.update_bscan(self.canvas.parser, self.canvas.display_range, self.canvas.current_e, self.canvas.annotation_manager.get_annotations())
        
        # 计算矩形的边界
        min_x = min(self.start_x, self.end_x)
        max_x = max(self.start_x, self.end_x)
        min_y = min(self.start_y, self.end_y)
        max_y = max(self.start_y, self.end_y)
        
        # 绘制矩形框
        self.axes.plot(
            [min_x, max_x, max_x, min_x, min_x],
            [min_y, min_y, max_y, max_y, min_y],
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        
        # 更新画布
        self.canvas.draw()
    
    def select_points_by_rect(self, x1, y1, x2, y2, add_points=True):
        """通过矩形框选择多个点
        
        Args:
            x1, y1: 矩形左上角坐标
            x2, y2: 矩形右下角坐标
            add_points: 是否添加选中的点，False则移除选中的点
        """
        # 计算矩形的边界
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        # 找到所有在矩形内的点
        selected_points = []
        
        # 获取当前显示的所有点
        displayed_points = self.canvas.displayed_points
        
        for point in displayed_points:
            point_x = point['e']
            point_y = point['depth']
            
            # 检查点是否在矩形内
            if min_x <= point_x <= max_x and min_y <= point_y <= max_y:
                selected_points.append(point)
        
        if add_points:
            # 左键框选：添加新选中的点
            # 如果不是多选模式，先清空选择
            if not self.is_multi_select:
                self.selected_points.clear()
            
            # 添加新选中的点，避免重复
            for point in selected_points:
                # 检查点是否已经被选中
                point_already_selected = False
                for selected_point in self.selected_points:
                    if self.points_equal(point, selected_point):
                        point_already_selected = True
                        break
                
                if not point_already_selected:
                    self.selected_points.append(point)
        else:
            # 右键框选：移除选中的点
            # 创建一个新的列表，包含不在矩形内或未被选中的点
            new_selected_points = []
            for selected_point in self.selected_points:
                # 检查点是否在矩形内
                point_in_rect = False
                for point in selected_points:
                    if self.points_equal(selected_point, point):
                        point_in_rect = True
                        break
                
                # 如果点不在矩形内，保留在选中列表中
                if not point_in_rect:
                    new_selected_points.append(selected_point)
            
            # 更新选中点列表
            self.selected_points = new_selected_points
        
        # 高亮显示选中的点
        self.highlight_selected_points()
        
        # 调用回调函数，更新选中点信息显示
        if self.callback:
            self.callback()
    
    def find_closest_point(self, x, y, threshold=10):
        """找到最近的点"""
        if not self.canvas.displayed_points:
            return None
        
        closest_point = None
        min_distance = float('inf')
        
        # 遍历所有显示的点，找到最近的点
        for p in self.canvas.displayed_points:
            point_x = p['e']
            point_y = p['depth']
            
            # 计算距离
            distance = np.sqrt((x - point_x) ** 2 + (y - point_y) ** 2)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = p
        
        # 如果距离小于阈值，返回该点
        if min_distance <= threshold:
            return closest_point
        else:
            return None
    
    def highlight_selected_points(self):
        """高亮显示选中的点"""
        # 设置画布的选中点
        self.canvas.selected_points = self.selected_points

        # 重新绘制画布（这会触发BScanCanvas的highlight_selected_points）
        self.canvas.update_bscan(self.canvas.parser, self.canvas.display_range, self.canvas.current_e, self.canvas.annotation_manager.get_annotations())
    
    def set_selected_points(self, points):
        """
        设置选中的点

        Args:
            points: 要选中的点列表，每个点应该是包含 'e', 'depth', 'amplitude' 等字段的字典
        """
        self.selected_points = list(points)  # 复制列表
        self.highlight_selected_points()

    def clear_selection(self):
        """清空选择"""
        self.selected_points.clear()
        # 保持多选模式，不重置
        # self.is_multi_select = False
        self.highlight_selected_points()
    
    def get_selected_points(self):
        """获取选中的点"""
        return self.selected_points
    
    def start_selection(self):
        """开始选择模式"""
        self.is_selecting = True
        # 显示提示信息
        print("进入选择模式：点击选择单个点，按住Shift键多选")
    
    def stop_selection(self):
        """停止选择模式"""
        self.is_selecting = False
        self.clear_selection()
        # 显示提示信息
        print("退出选择模式")