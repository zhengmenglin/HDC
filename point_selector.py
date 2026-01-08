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
    
    def __init__(self, canvas):
        """
        初始化点选择器
        
        Args:
            canvas: BScanCanvas实例
        """
        self.canvas = canvas
        self.axes = canvas.axes
        
        # 选中的点
        self.selected_points = []
        
        # 点的选择状态
        self.is_selecting = False
        self.is_multi_select = False
        
        # 矩形选择器
        self.rect_selector = None
        
        # 连接鼠标事件
        self.connect_events()
    
    def connect_events(self):
        """连接鼠标事件"""
        # 鼠标点击事件
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        
        # 键盘事件
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
    
    def on_button_press(self, event):
        """鼠标按下事件处理"""
        # 只处理左键点击
        if event.button != 1:  # 左键是1，右键是3
            return
        
        # 检查是否在坐标轴范围内
        if not event.inaxes == self.axes:
            return
        
        # 记录点击位置
        self.click_x = event.xdata
        self.click_y = event.ydata
    
    def on_button_release(self, event):
        """鼠标释放事件处理"""
        # 只处理左键点击
        if event.button != 1:  # 左键是1，右键是3
            return
        
        # 检查是否在坐标轴范围内
        if not event.inaxes == self.axes:
            return
        
        # 计算点击位置和释放位置的距离
        release_x = event.xdata
        release_y = event.ydata
        
        distance = np.sqrt((release_x - self.click_x) ** 2 + (release_y - self.click_y) ** 2)
        
        # 如果距离很小，认为是单击
        if distance < 5:  # 阈值可以调整
            self.select_single_point(self.click_x, self.click_y)
        else:
            # 否则认为是拖拽，进行框选
            self.select_points_by_rect(self.click_x, self.click_y, release_x, release_y)
    
    def on_key_press(self, event):
        """键盘按下事件处理"""
        if event.key in ['shift', 'ctrl']:
            self.is_multi_select = True
        elif event.key == 'escape':
            # 取消选择
            self.clear_selection()
    
    def on_key_release(self, event):
        """键盘释放事件处理"""
        if event.key in ['shift', 'ctrl']:
            self.is_multi_select = False
    
    def select_single_point(self, x, y):
        """选择单个点"""
        # 找到最近的点
        closest_point = self.find_closest_point(x, y)
        
        if closest_point:
            # 如果不是多选模式，先清空选择
            if not self.is_multi_select:
                self.selected_points.clear()
            
            # 如果点已经被选中，取消选择
            if closest_point in self.selected_points:
                self.selected_points.remove(closest_point)
            else:
                # 否则添加到选择列表
                self.selected_points.append(closest_point)
            
            # 高亮显示选中的点
            self.highlight_selected_points()
    
    def select_points_by_rect(self, x1, y1, x2, y2):
        """通过矩形框选择多个点"""
        # 计算矩形的边界
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        
        # 找到所有在矩形内的点
        selected = []
        
        # 获取当前显示的所有点
        # 注意：这里需要从canvas中获取当前显示的点
        # 由于当前的BScanCanvas没有保存绘制的点的信息，我们需要修改BScanCanvas来保存这些信息
        # 这里先简单实现，后续需要修改BScanCanvas
        
        # 如果不是多选模式，先清空选择
        if not self.is_multi_select:
            self.selected_points.clear()
        
        # TODO: 实现矩形框选功能
        # 需要修改BScanCanvas，保存绘制的点的信息
        
        # 高亮显示选中的点
        self.highlight_selected_points()
    
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
        # 重新绘制图像
        self.canvas.draw()
        
        # 绘制选中的点
        if self.selected_points:
            # 提取选中点的坐标
            x_vals = [p['e'] for p in self.selected_points]
            y_vals = [p['depth'] for p in self.selected_points]
            
            # 在画布上绘制选中的点，使用红色外圈高亮
            self.axes.scatter(x_vals, y_vals, 
                           c='none', 
                           s=100, 
                           edgecolors='red', 
                           linewidths=2, 
                           zorder=5)
            
            # 更新画布
            self.canvas.draw()
    
    def clear_selection(self):
        """清空选择"""
        self.selected_points.clear()
        self.is_multi_select = False
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