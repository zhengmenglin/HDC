#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钢轨探伤B显数据可视化主窗口
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider,
    QLabel, QMenuBar, QAction, QFileDialog, QMessageBox,
    QStatusBar, QSplitter, QTextEdit, QGroupBox, QComboBox,
    QPushButton, QRadioButton, QButtonGroup, QFrame, QCheckBox
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
from gui.type_selector import TypeSelectorDialog
from gui.point_selector import PointSelector
from gui.annotation_manager import AnnotationManager


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
        
        # 保存画布的属性，用于更新
        self.parser = None
        self.display_range = 255
        self.current_e = None
        self.annotation_manager = None
        
        # 自动缩放恢复标志，默认开启
        self.auto_reset_zoom = True
        
    def update_bscan(self, parser, display_range=255, current_e=None, annotations=None):
        """更新B显图像"""
        # 保存当前的轴范围，以便在更新后恢复（如果需要）
        # 无论auto_reset_zoom状态如何，都保存当前范围，以便用户使用工具栏缩放后能保持
        current_xlim = self.axes.get_xlim()
        current_ylim = self.axes.get_ylim()
        
        # 保存参数到画布属性
        self.parser = parser
        self.display_range = display_range
        self.current_e = current_e
        if annotations is not None:
            self.annotations = annotations
        
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
            
            return p['e'] - beam_offset / 3 + (p['depth'] * sin_theta) / 3
        
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
        
        # 绘制标注框
        if annotations:
            for annotation in annotations:
                points = annotation['points']
                if not points:
                    continue
                
                # 计算标注框的边界
                e_values = []
                depth_values = []
                for p in points:
                    adjusted_e = get_adjusted_e(p)
                    
                    # 获取通道角度
                    if p['channel'] in parser.params['beam_angle']:
                        angle = parser.params['beam_angle'][p['channel']]
                    else:
                        angle = parser.channel_config.get(p['channel'], {}).get('angle', 0)
                    
                    # 将角度转换为弧度
                    angle_rad = math.radians(angle)
                    cos_theta = math.cos(angle_rad)
                    
                    # 计算调整后的深度值
                    adjusted_depth = p['depth'] * cos_theta
                    
                    e_values.append(adjusted_e)
                    depth_values.append(adjusted_depth)
                
                # 获取边界
                min_e = min(e_values)
                max_e = max(e_values)
                min_depth = min(depth_values)
                max_depth = max(depth_values)
                
                # 绘制灰色框
                self.axes.plot(
                    [min_e, max_e, max_e, min_e, min_e],
                    [min_depth, min_depth, max_depth, max_depth, min_depth],
                    color='gray',
                    linestyle='-',
                    linewidth=2,  # 增加线宽，使其更明显
                    alpha=0.8
                )
                
                # 绘制灰色文字（标注名）在框的下方
                annotation_name = f"{annotation['subtype']['name']}"
                self.axes.text(
                    (min_e + max_e) / 2,  # 文字水平居中
                    max_depth + 5,  # 文字在框下方
                    annotation_name,
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='gray',
                    weight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3')
                )
        
        # 绘制选中的点
        self.highlight_selected_points()
        
        # 如果没有开启自动缩放恢复，恢复到用户之前的缩放状态
        # 这将保持用户使用matplotlib工具栏进行的缩放
        if not self.auto_reset_zoom:
            self.axes.set_xlim(current_xlim)
            self.axes.set_ylim(current_ylim)
        
        # 更新画布
        self.draw()
    
    def highlight_selected_points(self):
        """高亮显示选中的点"""
        if self.selected_points:
            # 提取选中点的坐标
            x_vals = [p['e'] for p in self.selected_points]
            y_vals = [p['depth'] for p in self.selected_points]
            
            # 绘制选中的点，使用橙色外圈高亮
            self.axes.scatter(x_vals, y_vals,
                           c='none',
                           s=150,  # 保持大小不变
                           edgecolors='orange',
                           linewidths=2,  # 减小线宽，使其更细
                           zorder=20)  # 保持zorder不变，确保在最上层
    
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
        
        # 设置画布的annotation_manager属性
        self.bscan_canvas.annotation_manager = self.annotation_manager
        
        # 添加NavigationToolbar
        self.toolbar = NavigationToolbar(self.bscan_canvas, self)
        self.main_layout.addWidget(self.toolbar)
        
        self.main_layout.addWidget(self.bscan_canvas)

        # 初始化点选择器
        self.point_selector = PointSelector(self.bscan_canvas, self.update_selected_points_display)

        # 创建选中点和已标注结构/伤损的并排显示区域
        self.create_combined_display()

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
                
                # 重置标注管理器
                self.annotation_manager.clear_annotations()
                
                # 尝试加载现有的标注文件
                import os
                json_dir = os.path.dirname(file_path)
                json_filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(json_filename)[0]
                annotation_filename = f"{name_without_ext}标注.json"
                annotation_file_path = os.path.join(json_dir, annotation_filename)
                
                if os.path.exists(annotation_file_path):
                    self.annotation_manager.load_annotations(annotation_file_path)
                    self.status_bar.showMessage(
                        f"已加载标注文件: {annotation_filename}", 3000
                    )
                
                # 更新B显画布，传递标注数据
                annotations = self.annotation_manager.get_annotations()
                self.bscan_canvas.update_bscan(self.parser, annotations=annotations)
                
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

                # 重置标注UI
                self.reset_annotation_ui()
                
                # 更新标注列表
                self.update_annotations_list()

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
    
    def on_damage_type_changed(self, index):
        """类型选择变化事件"""
        self.subtype_combo.clear()
        self.subtype_combo.setEnabled(False)
        self.save_annotation_btn.setEnabled(False)

        if index <= 0:  # 请选择类型
            return

        selected_type = self.damage_combo.currentText()
        type_data = self.damage_combo.currentData()
        self.subtype_combo.setEnabled(True)

        # 根据选择的类型设置子类型
        if type_data == "damage":  # 伤损类型
            if selected_type == "伤损":
                subtypes = [
                    ("1", "轨头外侧核伤"),
                    ("2", "轨头中心核伤"),
                    ("3", "轨头内侧核伤"),
                    ("4", "轨头内、外、中心核伤"),
                    ("5", "轨头内、中心核伤"),
                    ("6", "轨头外、中心核伤"),
                    ("7", "轨头水平分裂"),
                    ("8", "轨头、轨腰分裂"),
                    ("9", "轨腰水平裂"),
                    ("10", "轨底断裂"),
                    ("11", "螺孔裂纹"),
                    ("12", "其他伤损")
                ]
        elif type_data == "structure":  # 正常结构类型
            if selected_type == "正常结构":
                subtypes = [
                    ("101", "螺孔"),
                    ("102", "轨头"),
                    ("103", "轨腰"),
                    ("104", "轨底"),
                    ("105", "其他结构")
                ]
        else:
            return

        self.subtype_combo.addItem("请选择具体类型", "")
        for subtype_id, subtype_name in subtypes:
            self.subtype_combo.addItem(f"{subtype_name}", subtype_id)

        # 连接子类型选择变化事件
        self.subtype_combo.currentIndexChanged.connect(self.on_subtype_changed)

    def reset_annotation_ui(self):
        """重置标注UI"""
        self.damage_combo.setCurrentIndex(0)
        self.subtype_combo.clear()
        self.subtype_combo.setEnabled(False)
        self.save_annotation_btn.setEnabled(False)

    def on_subtype_changed(self, index):
        """子类型选择变化事件"""
        if index > 0 and self.point_selector.get_selected_points():
            self.save_annotation_btn.setEnabled(True)
        else:
            self.save_annotation_btn.setEnabled(False)

    def save_annotation_auto(self):
        """自动保存标注（不选择路径）"""
        if self.parser is None:
            QMessageBox.warning(self, "警告", "请先打开一个JSON文件")
            return

        # 获取选中的点
        selected_points = self.point_selector.get_selected_points()
        if not selected_points:
            QMessageBox.warning(self, "警告", "请先选择要标注的点")
            return

        # 获取选择的类型
        selected_type = self.damage_combo.currentText()
        type_data = self.damage_combo.currentData()
        subtype_index = self.subtype_combo.currentIndex()
        if subtype_index <= 0:
            QMessageBox.warning(self, "警告", "请选择具体类型")
            return

        subtype_text = self.subtype_combo.currentText()
        subtype_id = self.subtype_combo.currentData()

        # 创建标注类型信息
        annotation_type = "伤损" if type_data == "damage" else "结构"
        subtype = {
            'id': subtype_id,
            'name': subtype_text.split('. ', 1)[1] if '. ' in subtype_text else subtype_text
        }

        # 创建标注
        annotation = self.annotation_manager.create_annotation(
            annotation_type, subtype, [p['original'] for p in selected_points]
        )

        if annotation:
            # 自动生成保存路径
            import os
            json_file_path = self.parser.json_file
            json_dir = os.path.dirname(json_file_path)
            json_filename = os.path.basename(json_file_path)

            # 去掉扩展名，添加"标注"后缀
            name_without_ext = os.path.splitext(json_filename)[0]
            annotation_filename = f"{name_without_ext}标注.json"

            annotation_file_path = os.path.join(json_dir, annotation_filename)

            # 保存标注（AnnotationManager会自动合并到现有文件）
            if self.annotation_manager.save_annotations(annotation_file_path):
                QMessageBox.information(self, "成功", f"标注已保存到: {annotation_file_path}")
                # 清空选择和重置UI，但保持多选模式
                self.point_selector.clear_selection()
                self.damage_combo.setCurrentIndex(0)
                self.subtype_combo.clear()
                self.subtype_combo.setEnabled(False)
                self.save_annotation_btn.setEnabled(False)
                # 更新状态栏
                self.status_bar.showMessage("标注已保存")
                # 更新已标注列表，显示文件中的所有标注
                self.update_annotations_list()
                # 重新绘制B显，显示所有标注（包括新保存的和未保存的）
                annotations = self.annotation_manager.get_annotations()
                self.bscan_canvas.update_bscan(self.parser, self.display_range, self.current_e, annotations)
            else:
                QMessageBox.critical(self, "错误", "保存标注失败")

    def on_x_slider_changed(self, value):
        """X轴滑动条值变化时的处理函数"""
        # 滑动条从右向左拖动时，值减小，当前显示的编码值也减小
        self.current_e = value
        self.x_pos_label.setText(f"当前: {value}")

        if self.parser is not None:
            # 获取当前的标注数据
            annotations = self.annotation_manager.get_annotations()
            self.bscan_canvas.update_bscan(self.parser, self.display_range, self.current_e, annotations)



    def update_annotations_list(self):
        """更新标注列表"""
        if self.parser is None:
            return

        import os
        from gui.annotation_manager import AnnotationManager
        json_file_path = self.parser.json_file
        json_dir = os.path.dirname(json_file_path)
        json_filename = os.path.basename(json_file_path)

        # 去掉扩展名，添加"标注"后缀
        name_without_ext = os.path.splitext(json_filename)[0]
        annotation_filename = f"{name_without_ext}标注.json"
        annotation_file_path = os.path.join(json_dir, annotation_filename)

        # 清空当前标注列表
        self.annotations_list.clear()
        self.annotations_list.addItem("请选择标注", None)

        # 加载标注文件到临时管理器，避免影响内存中的未保存标注
        if os.path.exists(annotation_file_path):
            # 创建临时标注管理器，仅用于加载文件中的标注
            temp_manager = AnnotationManager()
            temp_manager.load_annotations(annotation_file_path)
            file_annotations = temp_manager.get_annotations()
            
            # 显示文件中的标注
            for annotation in file_annotations:
                annotation_name = f"{annotation['subtype']['name']} (E{annotation['start_E']}-E{annotation['end_E']})"
                self.annotations_list.addItem(annotation_name, annotation['id'])

    def on_annotation_selected(self, index):
        """当选择标注时更新详情"""
        if index <= 0:
            self.annotation_details_text.setPlainText("暂无选中的标注")
            self.delete_annotation_btn.setEnabled(False)
            return

        # 获取选中的标注ID
        annotation_id = self.annotations_list.currentData()
        annotation = self.annotation_manager.get_annotation_by_id(annotation_id)

        if annotation:
            # 构建详情文本
            details_lines = []
            details_lines.append(f"标注ID: {annotation['id']}")
            details_lines.append(f"类型: {annotation['type']}")
            details_lines.append(f"具体类型: {annotation['subtype']['name']}")
            details_lines.append(f"编码范围: E{annotation['start_E']} - E{annotation['end_E']}")
            details_lines.append(f"涉及通道: {annotation['channels']}")
            details_lines.append(f"点数: {len(annotation['points'])}")
            details_lines.append(f"创建时间: {annotation['timestamp']}")
            details_lines.append("")
            details_lines.append("包含的点:")
            
            for i, point in enumerate(annotation['points'], 1):
                details_lines.append(
                    f"{i}. E:{point['e']} Ch:{point['channel']} "
                    f"深度:{point['depth']:.1f} 幅度:{point['amplitude']:.1f}"
                )

            # 设置详情文本
            self.annotation_details_text.setPlainText("\n".join(details_lines))
            self.delete_annotation_btn.setEnabled(True)
        else:
            self.annotation_details_text.setPlainText("暂无选中的标注")
            self.delete_annotation_btn.setEnabled(False)
    
    def delete_selected_annotation(self):
        """删除选中的标注"""
        # 获取选中的标注ID
        annotation_id = self.annotations_list.currentData()
        if not annotation_id:
            return
        
        # 删除标注
        if self.annotation_manager.delete_annotation(annotation_id):
            # 保存修改后的标注到文件
            import os
            json_file_path = self.parser.json_file
            json_dir = os.path.dirname(json_file_path)
            json_filename = os.path.basename(json_file_path)
            name_without_ext = os.path.splitext(json_filename)[0]
            annotation_filename = f"{name_without_ext}标注.json"
            annotation_file_path = os.path.join(json_dir, annotation_filename)
            
            self.annotation_manager.save_annotations(annotation_file_path)
            
            # 更新标注列表
            self.update_annotations_list()
            
            # 重新绘制B显，显示更新后的标注
            annotations = self.annotation_manager.get_annotations()
            self.bscan_canvas.update_bscan(self.parser, self.display_range, self.current_e, annotations)
            
            QMessageBox.information(self, "成功", "标注已删除")
        else:
            QMessageBox.critical(self, "错误", "删除标注失败")
    
    def on_rect_select_state_changed(self, state):
        """处理框选开关状态变化"""
        # 设置点选择器的框选功能状态
        enabled = state == 2  # 2表示选中，0表示未选中
        self.point_selector.set_rect_select_enabled(enabled)
        
    def on_auto_zoom_reset_changed(self, state):
        """处理自动缩放恢复开关状态变化"""
        # 设置B显画布的自动缩放恢复状态
        enabled = state == 2  # 2表示选中，0表示未选中
        self.bscan_canvas.auto_reset_zoom = enabled

    def create_combined_display(self):
        """创建选中点和已标注结构/伤损的并排显示区域"""
        # 创建一个水平布局来容纳两个区域
        combined_layout = QHBoxLayout()
        
        # 左侧：选中点和标注控制面板
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 创建选中点信息显示区域
        group_box1 = QGroupBox("选中的点和标注")
        group_layout1 = QHBoxLayout(group_box1)
        
        # 左侧：选中点信息显示
        left_layout1 = QVBoxLayout()
        
        # 创建文本编辑器用于显示选中点信息
        self.selected_points_text = QTextEdit()
        self.selected_points_text.setReadOnly(True)
        self.selected_points_text.setMaximumHeight(120)  # 设置最大高度，大约5行
        self.selected_points_text.setPlainText("暂无选中的点")
        
        left_layout1.addWidget(QLabel("选中的点："))
        left_layout1.addWidget(self.selected_points_text)
        
        # 添加框选开关
        self.rect_select_checkbox = QCheckBox("启用框选功能")
        self.rect_select_checkbox.setChecked(True)  # 默认开启
        self.rect_select_checkbox.stateChanged.connect(self.on_rect_select_state_changed)
        left_layout1.addWidget(self.rect_select_checkbox)
        
        # 添加自动缩放恢复开关
        self.auto_zoom_reset_checkbox = QCheckBox("启用自动缩放恢复")
        self.auto_zoom_reset_checkbox.setChecked(True)  # 默认开启
        self.auto_zoom_reset_checkbox.stateChanged.connect(self.on_auto_zoom_reset_changed)
        left_layout1.addWidget(self.auto_zoom_reset_checkbox)
        
        # 右侧：标注控制面板
        right_layout1 = QVBoxLayout()
        
        # 伤损/结构类型选择
        type_label = QLabel("类型：")
        self.damage_combo = QComboBox()
        self.damage_combo.addItem("请选择类型", "")
        self.damage_combo.addItem("伤损", "damage")
        self.damage_combo.addItem("正常结构", "structure")
        
        # 子类型选择（根据选择的类型动态更新）
        subtype_label = QLabel("具体类型：")
        self.subtype_combo = QComboBox()
        self.subtype_combo.setMinimumWidth(200)  # 增加最小宽度，确保选项显示完整
        self.subtype_combo.setEnabled(False)
        
        # 连接类型选择变化事件
        self.damage_combo.currentIndexChanged.connect(self.on_damage_type_changed)
        
        # 保存标注按钮
        self.save_annotation_btn = QPushButton("保存标注")
        self.save_annotation_btn.clicked.connect(self.save_annotation_auto)
        self.save_annotation_btn.setEnabled(False)
        
        right_layout1.addWidget(type_label)
        right_layout1.addWidget(self.damage_combo)
        right_layout1.addWidget(subtype_label)
        right_layout1.addWidget(self.subtype_combo)
        right_layout1.addWidget(self.save_annotation_btn)
        right_layout1.addStretch()  # 添加伸缩空间
        
        # 添加分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.VLine)
        line1.setFrameShadow(QFrame.Sunken)
        
        group_layout1.addLayout(left_layout1)
        group_layout1.addWidget(line1)
        group_layout1.addLayout(right_layout1)
        
        left_layout.addWidget(group_box1)
        
        # 右侧：已标注结构/伤损显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        group_box2 = QGroupBox("已标注的结构/伤损")
        group_layout2 = QVBoxLayout(group_box2)  # 改为垂直布局
        
        # 上侧：标注列表
        self.annotations_list_label = QLabel("标注列表：")
        group_layout2.addWidget(self.annotations_list_label)
        
        # 创建标注列表视图，设置宽度
        self.annotations_list = QComboBox()
        self.annotations_list.setMinimumWidth(300)  # 增加宽度
        self.annotations_list.setMaximumHeight(120)
        self.annotations_list.currentIndexChanged.connect(self.on_annotation_selected)
        group_layout2.addWidget(self.annotations_list)
        
        # 添加刷新按钮和删除按钮
        button_layout = QHBoxLayout()
        self.refresh_annotations_btn = QPushButton("刷新标注列表")
        self.refresh_annotations_btn.clicked.connect(self.update_annotations_list)
        button_layout.addWidget(self.refresh_annotations_btn)
        
        self.delete_annotation_btn = QPushButton("删除选中标注")
        self.delete_annotation_btn.clicked.connect(self.delete_selected_annotation)
        self.delete_annotation_btn.setEnabled(False)
        button_layout.addWidget(self.delete_annotation_btn)
        group_layout2.addLayout(button_layout)
        
        # 下侧：标注详情
        self.annotation_details_label = QLabel("标注详情：")
        group_layout2.addWidget(self.annotation_details_label)
        
        # 创建标注详情文本框
        self.annotation_details_text = QTextEdit()
        self.annotation_details_text.setReadOnly(True)
        self.annotation_details_text.setMaximumHeight(120)
        self.annotation_details_text.setPlainText("暂无选中的标注")
        group_layout2.addWidget(self.annotation_details_text)
        
        right_layout.addWidget(group_box2)
        
        # 添加分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.VLine)
        line2.setFrameShadow(QFrame.Sunken)
        
        combined_layout.addWidget(left_widget, 1)  # 左侧占1份宽度
        combined_layout.addWidget(line2)
        combined_layout.addWidget(right_widget, 1)  # 右侧占1份宽度
        
        # 将组合布局添加到主布局
        self.main_layout.addLayout(combined_layout)
    
    def update_selected_points_display(self):
        """更新选中点信息显示"""
        selected_points = self.point_selector.get_selected_points()

        if not selected_points:
            self.selected_points_text.setPlainText("暂无选中的点")
            return

        # 构建显示文本
        display_lines = []
        for i, point in enumerate(selected_points, 1):
            original = point['original']
            display_lines.append(
                f"{i}. E:{original['e']} Ch:{original['channel']} "
                f"深度:{original['depth']:.1f} 幅度:{original['amplitude']:.1f}"
            )

        # 设置文本内容
        self.selected_points_text.setPlainText("\n".join(display_lines))
