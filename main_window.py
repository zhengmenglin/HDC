#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钢轨探伤B显数据可视化主窗口
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, 
    QLabel, QMenuBar, QAction, QFileDialog, QMessageBox,
    QStatusBar, QSplitter
)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')  # 使用Qt5后端

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = ['sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# 导入自定义模块
from type_selector import TypeSelectorDialog
from point_selector import PointSelector
from annotation_manager import AnnotationManager


class FlawDetectionDataParser:
    """钢轨探伤B显数据JSON解析器"""
    
    def __init__(self, json_file: str):
        """
        初始化解析器
        
        Args:
            json_file: JSON文件路径
        """
        self.json_file = json_file
        self.data = self._load_json()
        
        # 事件数据存储
        self.events = []  # 按顺序存储所有事件
        self.encoding_count = 0  # 编码计数器
        
        # B显数据存储
        self.b1_points = []  # 第一B显点 [{e, channel, depth, amplitude}, ...]
        self.b2_points = []  # 第二B显点 [{e, channel, depth, amplitude}, ...]
        
        # 系统参数
        self.params = {
            'sensitivity': {},      # 灵敏度 {channel: dB值}
            'gate_start': {},       # 闸门前沿 {channel: %}
            'gate_end': {},         # 闸门后沿 {channel: %}
            'gate_level': {},       # 报警阈值 {channel: %}
            'railbase_start': {},   # 轨底闸门前沿 {channel: %}
            'railbase_end': {},     # 轨底闸门后沿 {channel: %}
            'signalloss_level': {}, # 穿透报警阈值 {channel: %}
            'bolthole_start': {},   # 螺孔闸门前沿 {channel: %}
            'bolthole_end': {},     # 螺孔闸门后沿 {channel: %}
            'test_range': {},       # 探测范围 {channel: mm}
            'zero_delay': {},       # 零点延迟 {channel: us}
            'beam_angle': {},       # 声束角度 {channel: 度}
            'beam_way': {},         # 射向 {channel: 编号}
            'beam_offset': {},      # 拼孔值 {channel: mm}
            'tvg': {}               # 补偿曲线 {channel: [20个补偿点]}
        }
        
        # 作业信息
        self.job_info = {
            'bureau': '',           # 路局
            'division': '',         # 站段
            'workshop': '',         # 车间
            'area': '',             # 工区
            'team': '',             # 班组
            'operator': '',         # 执机人
            'device_num': '',       # 机号
            'device_type': '',      # 机型
            'date': '',             # 作业日期
            'time': '',             # 作业时间
            'location': '',         # 里程
            'updown': '',           # 上下行
            'leftright': '',        # 股别
            'speed': 0.0,           # 速度
            'suppress': '',         # 抑制
            'alarm': '',            # 报警声音
            'battery': '',          # 电池电量
        }
        
        # 通道配置映射：0-8代表1-9通道
        self.channel_config = {
            0: {'angle': 70, 'way': '前外', 'side': '右'},
            1: {'angle': -70, 'way': '后外', 'side': '右'},
            2: {'angle': 70, 'way': '前内', 'side': '右'},
            3: {'angle': -70, 'way': '后内', 'side': '右'},
            4: {'angle': 70, 'way': '前直', 'side': '右'},
            5: {'angle': -70, 'way': '后直', 'side': '右'},
            6: {'angle': 37, 'way': '前直', 'side': '右'},
            7: {'angle': -37, 'way': '后直', 'side': '右'},
            8: {'angle': 0, 'way': '直打', 'side': '右'},
        }
        
        # 解析数据
        self._parse()
    
    def _load_json(self) -> List[Dict]:
        """加载JSON文件，支持两种格式：
        1. JSON Lines格式（每行一个JSON对象）- 如123.json
        2. JSON数组格式（整个文件是一个数组）- 如*.cap.json
        """
        with open(self.json_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        events = []
        
        # 格式1: JSON Lines格式（每行一个JSON对象）
        if content.startswith('[') and content.endswith(']') and '\n' in content:
            # 可能是JSON数组格式（整个文件是一个数组）
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    events = data
                    print(f"检测到JSON数组格式，共{len(events)}个事件")
                    return events
            except json.JSONDecodeError:
                pass
        
        # 格式2: JSON Lines格式（每行一个JSON对象）
        if '\n' in content:
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 跳过JSON数组的括号
                line = line.strip(',[]')
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if isinstance(event, dict):
                        events.append(event)
                except json.JSONDecodeError:
                    # 尝试提取JSON对象
                    try:
                        start = line.find('{')
                        end = line.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            json_str = line[start:end+1]
                            event = json.loads(json_str)
                            if isinstance(event, dict):
                                events.append(event)
                    except:
                        pass
            if events:
                print(f"检测到JSON Lines格式，共{len(events)}个事件")
                return events
        
        # 格式3: 单行JSON数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                events = data
            elif isinstance(data, dict):
                events = [data]
            if events:
                print(f"检测到单行JSON格式，共{len(events)}个事件")
        except json.JSONDecodeError:
            pass
        
        return events
    
    def _parse(self):
        """解析所有事件"""
        for event in self.data:
            self._parse_event(event)
    
    def _parse_event(self, event: Dict):
        """解析单个事件"""
        if not isinstance(event, dict):
            return
        
        # 存储原始事件
        self.events.append(event)
        
        for key, value in event.items():
            if key == 'E':
                # 间隔编码
                self.encoding_count = int(value) if value is not None else 0
            
            elif key == 'B1':
                # 第一B显点 [channel, depth, amplitude, ...]
                if isinstance(value, list) and len(value) >= 3:
                    channel = int(value[0]) if value[0] is not None else 0
                    depth = float(value[1]) if value[1] is not None else 0.0
                    amplitude = float(value[2]) if value[2] is not None else 100.0
                    self.b1_points.append({
                        'e': self.encoding_count,
                        'channel': channel,
                        'depth': depth,
                        'amplitude': amplitude
                    })
            
            elif key == 'B2':
                # 第二B显点 [channel, depth, amplitude, ...]
                if isinstance(value, list) and len(value) >= 3:
                    channel = int(value[0]) if value[0] is not None else 0
                    depth = float(value[1]) if value[1] is not None else 0.0
                    amplitude = float(value[2]) if value[2] is not None else 100.0
                    self.b2_points.append({
                        'e': self.encoding_count,
                        'channel': channel,
                        'depth': depth,
                        'amplitude': amplitude
                    })
            
            # 系统参数解析...（省略，与原代码相同）
            elif key == 'Sensitivity':
                if isinstance(value, list):
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            sensitivity_val = value[i + 1]
                            self.params['sensitivity'][channel] = sensitivity_val
            
            elif key == 'BeamAngle':
                if isinstance(value, list):
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            beam_angle_val = value[i + 1]
                            
                            # 根据通道号将后向通道的角度转换为负数
                            # 通道1、3、5、7为后向通道
                            if channel in [1, 3, 5, 7]:
                                # 将后向通道的角度转换为负数
                                beam_angle_val = -abs(beam_angle_val)
                            
                            self.params['beam_angle'][channel] = beam_angle_val
            
            elif key == 'BeamOffset':
                if isinstance(value, list):
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            offset = value[i + 1]
                            self.params['beam_offset'][channel] = offset
            
            # 作业信息解析...（省略，与原代码相同）
            elif key == 'Bureau':
                self.job_info['bureau'] = str(value)
            elif key == 'Operator':
                self.job_info['operator'] = str(value)
            elif key == 'DeviceNum':
                self.job_info['device_num'] = str(value)
            elif key == 'Date':
                self.job_info['date'] = str(value)
            elif key == 'Time':
                self.job_info['time'] = str(value)
    
    def get_all_channels(self) -> List[int]:
        """获取所有有数据的通道"""
        channels = set(p['channel'] for p in self.b1_points)
        channels.update(p['channel'] for p in self.b2_points)
        return sorted(channels)
    
    def get_summary(self) -> Dict:
        """获取数据摘要"""
        return {
            'total_events': len(self.events),
            'encoding_count': self.encoding_count,
            'b1_count': len(self.b1_points),
            'b2_count': len(self.b2_points),
            'channels': self.get_all_channels(),
            'channels_with_data': len(self.get_all_channels()),
            'job_info': self.job_info
        }


class BScanCanvas(FigureCanvas):
    """B显数据可视化画布"""
    
    def __init__(self, parent=None, width=12, height=10, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # 通道颜色配置
        self.channel_colors = {
            0: '#8B008B',  # 通道0（对应1通道）：深紫色 A 前外 70°
            1: '#90EE90',  # 通道1（对应2通道）：浅绿色 B 后外 70°
            2: '#A52A2A',  # 通道2（对应3通道）：棕色 C 前内 70°
            3: '#87CEEB',  # 通道3（对应4通道）：天蓝色 D 后内 70°
            4: '#000000',  # 通道4（对应5通道）：黑色 E 前直 70°
            5: '#DDA0DD',  # 通道5（对应6通道）：浅紫色 F 后直 70°
            6: '#006400',  # 通道6（对应7通道）：深绿色 G 前 37°
            7: '#00008B',  # 通道7（对应8通道）：深蓝色 H 后 37°
            8: '#FF0000'   # 通道8（对应9通道）：红色 I 直打 0°
        }
        
        super(BScanCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置画布背景色
        self.axes.set_facecolor('white')
        
        # 初始化选中的点
        self.selected_points = []
        
        # 保存绘制的点信息，用于点选择
        self.displayed_points = []
        
    def update_bscan(self, parser, display_range=255, current_e=None):
        """更新B显图像"""
        # 清空当前图像
        self.axes.clear()
        
        # 清空显示点信息
        self.displayed_points.clear()
        
        # 获取数据
        b1_points = parser.b1_points
        b2_points = parser.b2_points
        channels = parser.get_all_channels()
        
        # 绘制B1点
        for channel in channels:
            # 获取通道的灵敏度值，默认为0
            sensitivity = parser.params['sensitivity'].get(channel, 0)
            
            # 过滤掉幅度低于灵敏度的点
            channel_data = [p for p in b1_points if p['channel'] == channel and p['amplitude'] >= sensitivity]
            
            if channel_data:
                # 获取通道的拼孔值，默认为0
                beam_offset = parser.params['beam_offset'].get(channel, 0)
                
                # 获取通道角度，默认为0度
                if channel in parser.params['beam_angle']:
                    angle = parser.params['beam_angle'][channel]
                else:
                    angle = parser.channel_config.get(channel, {}).get('angle', 0)
                
                # 将角度转换为弧度
                angle_rad = math.radians(angle)
                cos_theta = math.cos(angle_rad)
                sin_theta = math.sin(angle_rad)
                
                # 计算调整后的编码值和深度值
                e_vals = [p['e'] - beam_offset / 3 + (p['depth'] * sin_theta) / 3 for p in channel_data]
                depth_vals = [p['depth'] * cos_theta for p in channel_data]
                amp_vals = [p['amplitude'] for p in channel_data]
                
                # 根据幅度设置点的大小
                sizes = [10 + amp * 0.5 for amp in amp_vals]
                
                # 使用通道特定颜色
                color = self.channel_colors.get(channel, '#808080')
                
                self.axes.scatter(e_vals, depth_vals, 
                               c=color, 
                               s=sizes, 
                               alpha=0.7,
                               label=f'B1 Ch{channel}')
                
                # 保存显示的点信息
                for i, p in enumerate(channel_data):
                    self.displayed_points.append({
                        'original': p,
                        'e': e_vals[i],
                        'depth': depth_vals[i],
                        'amplitude': amp_vals[i],
                        'b_type': 'B1'
                    })
        
        # 绘制B2点（使用三角形标记）
        for channel in channels:
            # 获取通道的灵敏度值，默认为0
            sensitivity = parser.params['sensitivity'].get(channel, 0)
            
            # 过滤掉幅度低于灵敏度的点
            channel_data = [p for p in b2_points if p['channel'] == channel and p['amplitude'] >= sensitivity]
            
            if channel_data:
                # 获取通道的拼孔值，默认为0
                beam_offset = parser.params['beam_offset'].get(channel, 0)
                
                # 获取通道角度，默认为0度
                if channel in parser.params['beam_angle']:
                    angle = parser.params['beam_angle'][channel]
                else:
                    angle = parser.channel_config.get(channel, {}).get('angle', 0)
                
                # 将角度转换为弧度
                angle_rad = math.radians(angle)
                cos_theta = math.cos(angle_rad)
                sin_theta = math.sin(angle_rad)
                
                # 计算调整后的编码值和深度值
                e_vals = [p['e'] - beam_offset / 3 + (p['depth'] * sin_theta) / 3 for p in channel_data]
                depth_vals = [p['depth'] * cos_theta for p in channel_data]
                amp_vals = [p['amplitude'] for p in channel_data]
                
                sizes = [10 + amp * 0.5 for amp in amp_vals]
                
                # 使用通道特定颜色
                color = self.channel_colors.get(channel, '#808080')
                
                self.axes.scatter(e_vals, depth_vals, 
                               c=color, 
                               s=sizes, 
                               marker='^',
                               alpha=0.7,
                               label=f'B2 Ch{channel}')
                
                # 保存显示的点信息
                for i, p in enumerate(channel_data):
                    self.displayed_points.append({
                        'original': p,
                        'e': e_vals[i],
                        'depth': depth_vals[i],
                        'amplitude': amp_vals[i],
                        'b_type': 'B2'
                    })
        
        # 设置主图属性 - 以右上角为原点
        self.axes.set_xlabel('编码间隔 (E)', fontsize=12)
        self.axes.set_ylabel('深度 (mm)', fontsize=12)
        self.axes.set_title('钢轨探伤B显数据可视化\n(右上角为原点, X向左递增, Y向下递增)', fontsize=14, fontweight='bold')
        self.axes.legend(loc='upper left', fontsize=8, ncol=2)
        self.axes.grid(True, alpha=0.3)
        
        # 计算所有调整后的编码值
        def get_adjusted_e(p):
            beam_offset = parser.params['beam_offset'].get(p['channel'], 0)
            
            if p['channel'] in parser.params['beam_angle']:
                angle = parser.params['beam_angle'][p['channel']]
            else:
                angle = parser.channel_config.get(p['channel'], {}).get('angle', 0)
            
            angle_rad = math.radians(angle)
            sin_theta = math.sin(angle_rad)
            
            return p['e'] - int(beam_offset / 3) + int((p['depth'] * sin_theta) / 3)
        
        adjusted_e_values = []
        for p in b1_points + b2_points:
            adjusted_e_values.append(get_adjusted_e(p))
        
        max_e = max(adjusted_e_values + [255], default=255)
        
        # 确定显示范围
        if current_e is None:
            current_e = max_e
        
        # 使用固定的X轴显示范围，与窗口宽度匹配
        x_range = display_range
        
        # 反转X轴，使E=0在右边，最大E在左边（X向左递增）
        self.axes.set_xlim(left=current_e + 10, right=current_e - x_range - 10)
        
        # Y轴设置，确保深度0在顶部，深度大值在底部（Y向下递增）
        # 保持右上角为原点
        self.axes.set_ylim(top=-5, bottom=200)  # 固定Y轴范围，确保原点在右上角
        
        # 取消固定的xy轴比例，允许Y轴自由缩放，但保持右上角为原点
        self.axes.set_aspect('auto')
        
        # 绘制选中的点
        self.highlight_selected_points()
        
        # 更新画布
        self.draw()
    
    def highlight_selected_points(self):
        """高亮显示选中的点"""
        if self.selected_points:
            # 提取选中点的坐标
            x_vals = [p['e'] for p in self.selected_points]
            y_vals = [p['depth'] for p in self.selected_points]
            
            # 绘制选中的点，使用红色外圈高亮
            self.axes.scatter(x_vals, y_vals, 
                           c='none', 
                           s=100, 
                           edgecolors='red', 
                           linewidths=2, 
                           zorder=5)
    
    def get_displayed_points(self):
        """获取显示的点列表"""
        return self.displayed_points


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("钢轨探伤B显数据可视化工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化数据解析器
        self.parser = None
        
        # 初始化当前显示的编码值范围
        self.current_e = None
        self.display_range = 255
        
        # 初始化标注管理器
        self.annotation_manager = AnnotationManager()
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建B显画布
        self.bscan_canvas = BScanCanvas(self, width=12, height=8)
        
        # 添加NavigationToolbar
        self.toolbar = NavigationToolbar(self.bscan_canvas, self)
        self.main_layout.addWidget(self.toolbar)
        
        self.main_layout.addWidget(self.bscan_canvas)
        
        # 初始化点选择器
        self.point_selector = PointSelector(self.bscan_canvas)
        
        # 创建X轴滑动条
        self.create_x_axis_slider()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 打开文件菜单项
        open_action = QAction("打开", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # 保存图像菜单项
        save_action = QAction("保存图像", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        # 退出菜单项
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 标注菜单
        annotation_menu = menubar.addMenu("标注")
        
        # 开始标注菜单项
        start_annotation_action = QAction("开始标注", self)
        start_annotation_action.triggered.connect(self.start_annotation)
        annotation_menu.addAction(start_annotation_action)
        
        # 保存标注菜单项
        save_annotation_action = QAction("保存标注", self)
        save_annotation_action.triggered.connect(self.save_annotation)
        annotation_menu.addAction(save_annotation_action)
        
        # 加载标注菜单项
        load_annotation_action = QAction("加载标注", self)
        load_annotation_action.triggered.connect(self.load_annotation)
        annotation_menu.addAction(load_annotation_action)
    
    def create_x_axis_slider(self):
        """创建X轴滑动条"""
        slider_layout = QHBoxLayout()
        
        # 滑动条标签
        slider_label = QLabel("X轴范围:")
        slider_layout.addWidget(slider_label)
        
        # 滑动条
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(255)
        self.x_slider.setValue(255)
        self.x_slider.setSingleStep(1)
        self.x_slider.setPageStep(10)
        self.x_slider.valueChanged.connect(self.on_x_slider_changed)
        slider_layout.addWidget(self.x_slider, 1)
        
        # 当前位置标签
        self.x_pos_label = QLabel("当前: 255")
        slider_layout.addWidget(self.x_pos_label)
        
        self.main_layout.addLayout(slider_layout)
    
    def open_file(self):
        """打开JSON文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开JSON文件", ".", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # 创建解析器
                self.parser = FlawDetectionDataParser(file_path)
                
                # 更新B显画布
                self.bscan_canvas.update_bscan(self.parser)
                
                # 获取摘要信息
                summary = self.parser.get_summary()
                self.status_bar.showMessage(
                    f"文件: {file_path} | 总事件数: {summary['total_events']} | B1点数: {summary['b1_count']} | B2点数: {summary['b2_count']} | 通道: {summary['channels']}"
                )
                
                # 更新滑动条范围
                max_e = self.parser.encoding_count
                self.x_slider.setMaximum(max_e)
                self.x_slider.setValue(max_e)
                self.x_pos_label.setText(f"当前: {max_e}")
                self.current_e = max_e
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"打开文件失败: {str(e)}")
    
    def save_image(self):
        """保存图像"""
        if self.parser is None:
            QMessageBox.warning(self, "警告", "请先打开一个JSON文件")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", ".", "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                self.bscan_canvas.fig.savefig(file_path, dpi=150, bbox_inches='tight')
                QMessageBox.information(self, "成功", f"图像已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")
    
    def start_annotation(self):
        """开始标注"""
        if self.parser is None:
            QMessageBox.warning(self, "警告", "请先打开一个JSON文件")
            return
        
        # 打开类型选择对话框
        dialog = TypeSelectorDialog(self)
        if dialog.exec_():
            # 获取选择结果
            selected = dialog.get_selected_type()
            annotation_type = selected['type']
            subtype = selected['subtype']
            
            # 开始选择点
            self.point_selector.start_selection()
            
            # 提示用户选择点
            QMessageBox.information(self, "标注", "请在图像上选择点，按住Shift键可多选，按Esc键取消选择")
            
            # 更新状态栏
            self.status_bar.showMessage(f"进入标注模式：{annotation_type} - {subtype['name']}")
    
    def save_annotation(self):
        """保存标注"""
        if self.parser is None:
            QMessageBox.warning(self, "警告", "请先打开一个JSON文件")
            return
        
        # 获取选中的点
        selected_points = self.point_selector.get_selected_points()
        if not selected_points:
            QMessageBox.warning(self, "警告", "请先选择要标注的点")
            return
        
        # 打开类型选择对话框
        dialog = TypeSelectorDialog(self)
        if dialog.exec_():
            # 获取选择结果
            selected = dialog.get_selected_type()
            annotation_type = selected['type']
            subtype = selected['subtype']
            
            # 创建标注
            annotation = self.annotation_manager.create_annotation(
                annotation_type, subtype, [p['original'] for p in selected_points]
            )
            
            if annotation:
                # 保存标注到文件
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "保存标注", ".", "JSON Files (*.json);;All Files (*)"
                )
                
                if file_path:
                    # 保存所有标注
                    if self.annotation_manager.save_annotations(file_path):
                        QMessageBox.information(self, "成功", f"标注已保存到: {file_path}")
                        # 清空选择
                        self.point_selector.clear_selection()
                        # 更新状态栏
                        self.status_bar.showMessage("标注已保存")
                    else:
                        QMessageBox.critical(self, "错误", "保存标注失败")
    
    def load_annotation(self):
        """加载标注"""
        if self.parser is None:
            QMessageBox.warning(self, "警告", "请先打开一个JSON文件")
            return
        
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载标注", ".", "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            # 加载标注
            if self.annotation_manager.load_annotations(file_path):
                # 获取标注数量
                annotation_count = len(self.annotation_manager.get_annotations())
                QMessageBox.information(self, "成功", f"已加载 {annotation_count} 个标注")
                # 更新状态栏
                self.status_bar.showMessage(f"已加载 {annotation_count} 个标注")
            else:
                QMessageBox.critical(self, "错误", "加载标注失败")
    
    def on_x_slider_changed(self, value):
        """X轴滑动条值变化时的处理函数"""
        # 滑动条从右向左拖动时，值减小，当前显示的编码值也减小
        self.current_e = value
        self.x_pos_label.setText(f"当前: {value}")
        
        if self.parser is not None:
            self.bscan_canvas.update_bscan(self.parser, self.display_range, self.current_e)
