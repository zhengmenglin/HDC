#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钢轨探伤B显数据JSON解析与可视化工具
根据《钢轨探伤B显数据JSON数据交换格式》规范编写

支持的事件类型：
- E: 间隔编码（横轴坐标）
- B1/B2: 第一/第二B显点 [通道号, 深度, 幅度]
- 系统参数: Sensitivity, GateStart/End, GateLevel, RailBaseStart/End等
- 作业信息: Bureau, Division, Operator, DeviceNum, Date, Time等
- 其他: Comment, Back, Remark
"""

import json
import sys
import os
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


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
        self.b1_points = []  # 第一B显点 [(E, channel, depth, amplitude), ...]
        self.b2_points = []  # 第二B显点 [(E, channel, depth, amplitude), ...]
        
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
                # 支持不同长度格式: [ch, depth, amp] 或 [ch, depth, amp, x, y]
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
            
            elif key == 'Sensitivity':
                # 灵敏度 [channel, dB值] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            sensitivity_val = value[i + 1]
                            self.params['sensitivity'][channel] = sensitivity_val
            
            elif key == 'GateStart':
                # 闸门前沿 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            gate_start_val = value[i + 1]
                            self.params['gate_start'][channel] = gate_start_val
            
            elif key == 'GateEnd':
                # 闸门后沿 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            gate_end_val = value[i + 1]
                            self.params['gate_end'][channel] = gate_end_val
            
            elif key == 'GateLevel':
                # 报警阈值 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            gate_level_val = value[i + 1]
                            self.params['gate_level'][channel] = gate_level_val
            
            elif key == 'RailBaseStart':
                # 轨底闸门前沿 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            railbase_start_val = value[i + 1]
                            self.params['railbase_start'][channel] = railbase_start_val
            
            elif key == 'RailBaseEnd':
                # 轨底闸门后沿 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            railbase_end_val = value[i + 1]
                            self.params['railbase_end'][channel] = railbase_end_val
            
            elif key == 'SignalLossLevel':
                # 穿透报警阈值 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            signalloss_level_val = value[i + 1]
                            self.params['signalloss_level'][channel] = signalloss_level_val
            
            elif key == 'BoltHoleStart':
                # 螺孔闸门前沿 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            bolthole_start_val = value[i + 1]
                            self.params['bolthole_start'][channel] = bolthole_start_val
            
            elif key == 'BoltHoleEnd':
                # 螺孔闸门后沿 [channel, %] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            bolthole_end_val = value[i + 1]
                            self.params['bolthole_end'][channel] = bolthole_end_val
            
            elif key == 'TestRange':
                # 探测范围 [channel, mm] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            test_range_val = value[i + 1]
                            self.params['test_range'][channel] = test_range_val
            
            elif key == 'ZeroDelay':
                # 零点延迟 [channel, us] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            zero_delay_val = value[i + 1]
                            self.params['zero_delay'][channel] = zero_delay_val
            
            elif key == 'BeamAngle':
                # 声束角度 [channel, 度] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
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
            
            elif key == 'BeamWay':
                # 射向 [channel, 编号] 或 [channel1, value1, channel2, value2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, value]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            beam_way_val = value[i + 1]
                            self.params['beam_way'][channel] = beam_way_val
            
            elif key == 'BeamOffset':
                # 拼孔值 [channel, mm] 或 [channel1, offset1, channel2, offset2, ...]
                if isinstance(value, list):
                    # 处理多通道格式：每两个元素为一组 [channel, offset]
                    for i in range(0, len(value), 2):
                        if i + 1 < len(value):
                            channel = value[i]
                            offset = value[i + 1]
                            self.params['beam_offset'][channel] = offset
            
            elif key == 'TVG':
                # 补偿曲线 [channel, [20个点]]
                if isinstance(value, list) and len(value) >= 2:
                    channel = value[0]
                    tvg_data = value[1] if isinstance(value[1], list) else []
                    self.params['tvg'][channel] = tvg_data
            
            # 作业信息
            elif key == 'Bureau':
                self.job_info['bureau'] = str(value)
            elif key == 'Division':
                self.job_info['division'] = str(value)
            elif key == 'Workshop':
                self.job_info['workshop'] = str(value)
            elif key == 'Area':
                self.job_info['area'] = str(value)
            elif key == 'Team':
                self.job_info['team'] = str(value)
            elif key == 'Operator':
                self.job_info['operator'] = str(value)
            elif key == 'DeviceNum':
                self.job_info['device_num'] = str(value)
            elif key == 'DeviceType':
                self.job_info['device_type'] = str(value)
            elif key == 'Date':
                self.job_info['date'] = str(value)
            elif key == 'Time':
                self.job_info['time'] = str(value)
            elif key == 'Location':
                self.job_info['location'] = str(value)
            elif key == 'UpDown':
                self.job_info['updown'] = str(value)
            elif key == 'LeftRight':
                self.job_info['leftright'] = str(value)
            elif key == 'Speed':
                self.job_info['speed'] = float(value) if value else 0.0
            elif key == 'Suppress':
                self.job_info['suppress'] = str(value)
            elif key == 'Alarm':
                self.job_info['alarm'] = str(value)
            elif key == 'Battery':
                self.job_info['battery'] = str(value)
    
    def get_b1_by_channel(self, channel: int) -> List[Tuple[int, float, float]]:
        """获取指定通道的B1点数据"""
        return [(p['e'], p['depth'], p['amplitude']) 
                for p in self.b1_points if p['channel'] == channel]
    
    def get_b2_by_channel(self, channel: int) -> List[Tuple[int, float, float]]:
        """获取指定通道的B2点数据"""
        return [(p['e'], p['depth'], p['amplitude']) 
                for p in self.b2_points if p['channel'] == channel]
    
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


def visualize_bscan(parser: FlawDetectionDataParser, output_file: str = None, show_plot: bool = True):
    """
    可视化B显数据
    
    Args:
        parser: FlawDetectionDataParser实例
        output_file: 输出文件路径（可选，如果不提供则自动生成）
        show_plot: 是否显示交互式图形（默认为True）
    """
    try:
        import matplotlib
        # 使用交互式后端以支持缩放
        matplotlib.use('TkAgg')  # 交互式后端，支持缩放
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except ImportError:
        print("错误: 需要安装matplotlib库")
        print("请运行: pip install matplotlib")
        return
    except Exception as e:
        # 如果TkAgg不可用，尝试使用其他交互式后端
        print(f"尝试使用TkAgg后端失败: {e}")
        try:
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
        except Exception as e2:
            print(f"尝试使用Qt5Agg后端失败: {e2}")
            print("将使用默认后端")
            import matplotlib.pyplot as plt
            from matplotlib import rcParams
    
    # 自动生成输出文件名
    if output_file is None:
        json_basename = os.path.splitext(os.path.basename(parser.json_file))[0]
        output_file = f"{json_basename}_bscan.png"
    
    # 设置中文字体，确保中文能正确显示
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    rcParams['font.family'] = ['sans-serif']
    rcParams['axes.unicode_minus'] = False
    # 为等宽字体设置中文支持
    rcParams['font.monospace'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans Mono']
    
    # 获取数据
    b1_points = parser.b1_points
    b2_points = parser.b2_points
    channels = parser.get_all_channels()
    
    # 设置图形尺寸，宽度略大于高度以容纳右侧信息面板
    fig = plt.figure(figsize=(12, 10))
    
    # 创建网格布局：主图左侧6列，右侧3列，增加主图区域高度
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 3], wspace=0.02)
    
    # 主图区域
    ax_main = fig.add_subplot(gs[0])
    
    # 为0-8通道设置特定颜色（对应1-9通道）
    channel_colors = {
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
    
    def plot_bscan_points(points, point_type, marker='o'):
        """绘制B显点的辅助函数

        Args:
            points: 点数据列表
            point_type: 点类型 ('B1' 或 'B2')
            marker: matplotlib标记类型
        """
        for channel in channels:
            # 获取通道的灵敏度值，默认为0
            sensitivity =0# parser.params['sensitivity'].get(channel, 0)

            # 过滤掉幅度低于灵敏度的点
            channel_data = [(p['e'], p['depth'], p['amplitude'])
                          for p in points if p['channel'] == channel and p['amplitude'] >= sensitivity]

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

                # 计算cos(θ)和sin(θ)
                cos_theta = math.cos(angle_rad)
                sin_theta = math.sin(angle_rad)

                # 应用拼孔值：编码值单位是3mm，拼孔值单位是1mm，所以应该减去1/3的拼孔值
                # 横向移动距离 = 编码值 - beam_offset/3 + (深度值×sin(θ))/3（深度单位是mm，编码值单位是3mm，需要转换）
                e_vals = [d[0] - beam_offset / 3 + (d[1] * sin_theta) / 3 for d in channel_data]
                # 深度值 = 原始深度×cos(θ)
                depth_vals = [d[1] * cos_theta for d in channel_data]
                amp_vals = [d[2] for d in channel_data]

                # 根据幅度设置点的大小
                sizes = [10 + amp * 0.3 for amp in amp_vals]
                if angle < 0:
                    # 如果是负数角度，使用统一大小
                    sizes = [10 for _ in amp_vals]

                # 使用通道特定颜色
                color = channel_colors.get(channel, '#808080')

                ax_main.scatter(e_vals, depth_vals,
                               c=color,
                               s=sizes,
                               marker=marker,
                               alpha=0.7,
                               label=f'{point_type} Ch{channel}')

    # 绘制B1点（圆点）和B2点（三角形）
    plot_bscan_points(b1_points, 'B1', 'o')
    plot_bscan_points(b2_points, 'B2', '^')

    
    # 设置主图属性 - 以右上角为原点
    ax_main.set_xlabel('编码间隔 (E)', fontsize=12)
    ax_main.set_ylabel('深度 (mm)', fontsize=12)
    ax_main.set_title('钢轨探伤B显数据可视化\n(右上角为原点, X向左递增, Y向下递增)', fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper left', fontsize=8, ncol=2)
    ax_main.grid(True, alpha=0.3)
    
    # 设置坐标系：右上角为原点
    # X轴：向右为负方向（E值从右向左递增），显示范围255个单位
    # Y轴：向下为正方向（深度值向下递增），显示范围255个单位
    
    # 计算调整后的X坐标（编码值 - 1/3拼孔值 + 深度×sin(θ)/3）的最大值
    def get_adjusted_e(p):
        beam_offset = parser.params['beam_offset'].get(p['channel'], 0)
        
        # 获取通道角度，默认为0度
        if p['channel'] in parser.params['beam_angle']:
            angle = parser.params['beam_angle'][p['channel']]
        else:
            angle = parser.channel_config.get(p['channel'], {}).get('angle', 0)
        
        # 将角度转换为弧度
        angle_rad = math.radians(angle)
        sin_theta = math.sin(angle_rad)
        
        # 编码值单位是3mm，拼孔值单位是1mm，所以应该减去1/3的拼孔值
        # 横向移动距离 = 编码值 - beam_offset/3 + (深度值×sin(θ))/3（深度单位是mm，编码值单位是3mm，需要转换）
        return p['e'] - beam_offset / 3 + (p['depth'] * sin_theta) / 3
    
    adjusted_e_values = []
    for p in b1_points + b2_points:
        adjusted_e_values.append(get_adjusted_e(p))
    
    max_e = max(adjusted_e_values + [255], default=255)
    
    # 设置固定显示范围：X轴和Y轴都是255个单位
    x_range = 255
    y_range = 255
    
    # 反转X轴，使E=0在右边，最大E在左边，显示范围为x_range个单位
    ax_main.set_xlim(left=max_e + 10, right=max_e - x_range - 10)
    
    # Y轴设置，使深度0在顶部，深度大值在底部，显示范围为y_range个单位
    ax_main.set_ylim(top=-5, bottom=y_range + 5)
    
    # 设置xy轴比例为1:1
    ax_main.set_aspect('equal', adjustable='box')
    
    # 右侧信息面板
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis('off')
    
    # 构建信息文本
    info_text = []
    
    # 作业信息
    info_text.append("=" * 40)
    info_text.append("作业信息")
    info_text.append("=" * 40)
    job = parser.job_info
    if job['device_num']:
        info_text.append(f"机号: {job['device_num']}")
    if job['device_type']:
        info_text.append(f"机型: {job['device_type']}")
    if job['operator']:
        info_text.append(f"执机人: {job['operator']}")
    if job['date']:
        info_text.append(f"日期: {job['date']}")
    if job['time']:
        info_text.append(f"时间: {job['time']}")
    if job['location']:
        info_text.append(f"里程: {job['location']} km")
    if job['updown']:
        info_text.append(f"行别: {job['updown']}")
    if job['leftright']:
        info_text.append(f"股别: {job['leftright']}")
    if job['speed']:
        info_text.append(f"速度: {job['speed']} km/h")
    
    # 通道统计
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("通道统计")
    info_text.append("=" * 40)
    info_text.append(f"总编码数: {parser.encoding_count}")
    info_text.append(f"B1点数: {len(b1_points)}")
    info_text.append(f"B2点数: {len(b2_points)}")
    info_text.append(f"活动通道: {', '.join(map(str, channels))}")
    
    # 通道配置
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("通道配置")
    info_text.append("=" * 40)
    for ch in channels[:9]:  # 显示前9个通道
        config = parser.channel_config.get(ch, {'angle': '?', 'way': '?', 'side': '?'})
        info_text.append(f"Ch{ch}: {config['angle']}° {config['way']} {config['side']}")
    
    # 灵敏度表
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("灵敏度 (dB)")
    info_text.append("=" * 40)
    sensitivity = parser.params['sensitivity']
    if sensitivity:
        for ch in sorted(sensitivity.keys())[:9]:
            info_text.append(f"Ch{ch}: {sensitivity[ch]:.1f} dB")
    
    # 闸门参数
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("闸门参数 (%)")
    info_text.append("=" * 40)
    gate_start = parser.params['gate_start']
    gate_end = parser.params['gate_end']
    if gate_start or gate_end:
        for ch in sorted(set(list(gate_start.keys()) + list(gate_end.keys())))[:9]:
            start = gate_start.get(ch, '-')
            end = gate_end.get(ch, '-')
            info_text.append(f"Ch{ch}: 前沿{start}% 后沿{end}%")
    
    # 探测范围
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("探测范围 (mm)")
    info_text.append("=" * 40)
    test_range = parser.params['test_range']
    if test_range:
        for ch in sorted(test_range.keys())[:9]:
            info_text.append(f"Ch{ch}: {test_range[ch]} mm")
    
    # 声束角度
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("声束角度 (°)")
    info_text.append("=" * 40)
    beam_angle = parser.params['beam_angle']
    if beam_angle:
        for ch in sorted(beam_angle.keys())[:9]:
            info_text.append(f"Ch{ch}: {beam_angle[ch]}°")
    
    # 拼孔值
    info_text.append("")
    info_text.append("=" * 40)
    info_text.append("拼孔值 (mm)")
    info_text.append("=" * 40)
    beam_offset = parser.params['beam_offset']
    if beam_offset:
        for ch in sorted(beam_offset.keys())[:9]:
            info_text.append(f"Ch{ch}: {beam_offset[ch]} mm")
    
    # 显示信息
    info_str = "\n".join(info_text)
    ax_info.text(0.05, 0.98, info_str, 
                transform=ax_info.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图像已保存到: {os.path.abspath(output_file)}")
    
    # 显示交互式图形
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """主函数"""
    import glob
    
    json_file = None
    output_file = None
    show_plot = True
    
    # 解析命令行参数
    if len(sys.argv) < 2:
        # 没有提供json文件路径，从当前目录找第一个json文件
        json_files = glob.glob("*.json")
        if json_files:
            json_file = json_files[0]
            print(f"未指定JSON文件，自动使用: {json_file}")
        else:
            print("错误: 当前目录没有找到JSON文件")
            print("用法: python parse_flaw_detection_json.py <json文件路径> [输出图像文件] [--no-show]")
            print("示例: python parse_flaw_detection_json.py 123.json")
            print("示例: python parse_flaw_detection_json.py 123.json output.png")
            print("示例: python parse_flaw_detection_json.py 123.json --no-show")
            sys.exit(1)
    else:
        json_file = sys.argv[1]
        # 解析其他参数
        for arg in sys.argv[2:]:
            if arg == '--no-show':
                show_plot = False
            else:
                output_file = arg
    
    if not os.path.exists(json_file):
        print(f"错误: 文件不存在 - {json_file}")
        sys.exit(1)
    
    print(f"正在解析: {json_file}")
    
    # 创建解析器
    parser = FlawDetectionDataParser(json_file)
    
    # 打印摘要
    summary = parser.get_summary()
    print(f"\n数据摘要:")
    print(f"  总事件数: {summary['total_events']}")
    print(f"  编码间隔数: {summary['encoding_count']}")
    print(f"  B1点数: {summary['b1_count']}")
    print(f"  B2点数: {summary['b2_count']}")
    print(f"  活动通道: {summary['channels']}")
    
    if summary['job_info']['device_num']:
        print(f"\n作业信息:")
        print(f"  机号: {summary['job_info']['device_num']}")
        print(f"  执机人: {summary['job_info']['operator']}")
        print(f"  日期: {summary['job_info']['date']}")
    
    # 可视化
    print(f"\n正在生成可视化图像...")
    visualize_bscan(parser, output_file, show_plot)


if __name__ == "__main__":
    main()
