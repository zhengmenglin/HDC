"""
坐标转换工具模块
提供超声波数据坐标转换功能，用于显示和聚类前调用
"""

import math
from typing import Dict, List, Tuple


class CoordinateConverter:
    """
    超声波数据坐标转换器
    提供坐标转换功能，用于将原始数据转换为显示和聚类所需的坐标
    """
    
    def __init__(self):
        # 通道配置映射：0-8代表A-I通道
        self.channel_config = {
            0: {'angle': 70, 'way': '后外', 'side': '右'},
            1: {'angle': 70, 'way': '前外', 'side': '右'},
            2: {'angle': 70, 'way': '后内', 'side': '左'},
            3: {'angle': 70, 'way': '前内', 'side': '左'},
            4: {'angle': 70, 'way': '后直', 'side': '右'},
            5: {'angle': 70, 'way': '前直', 'side': '右'},
            6: {'angle': 37, 'way': '后直', 'side': '右'},
            7: {'angle': 37, 'way': '前直', 'side': '右'},
            8: {'angle': 0, 'way': '直打', 'side': '右'},
        }
        
        # 轨型对应的最大深度映射
        self.max_depth_by_rail_type = {
            "45": 34,
            "50": 36,
            "60": 39,
            "75": 48
        }
    
    def calculate_adjusted_depth(self, channel: int, depth: float, display_mode: str) -> float:
        """
        根据显示模式计算调整后的深度，用于处理二次反射
        
        Args:
            channel: 通道号
            depth: 原始深度值
            display_mode: 显示模式，"V型"或"压缩"
            
        Returns:
            调整后的深度值
        """
        # 只有斜70°通道（0-3）需要特殊处理
        if not (0 <= channel <= 3):
            return depth
        
        if display_mode == "V型":
            # 方案1：根据轨型确定最大深度，处理二次反射
            # 这里使用默认的60轨最大深度39
            max_depth = self.max_depth_by_rail_type.get("60", 39)  # 默认60轨
            if depth > max_depth:
                adjusted_depth = max_depth + max_depth - depth
                return adjusted_depth
            return depth
        elif display_mode == "压缩":
            # 方案2：直接将深度*0.5
            return depth * 0.5
        return depth
    
    def calculate_display_coordinates(self, 
                                    e: int, 
                                    channel: int, 
                                    depth: float, 
                                    beam_offset: float, 
                                    beam_way: int = 0,
                                    beam_angle: float = None,
                                    display_mode: str = "压缩") -> Tuple[float, float]:
        """
        计算显示和聚类所需的坐标
        
        Args:
            e: 原始E值
            channel: 通道号
            depth: 原始深度值
            beam_offset: 拼孔值
            beam_way: 射向（-1后、0直、1前），默认为0
            beam_angle: 声束角度，默认从channel_config获取
            display_mode: 显示模式，"V型"或"压缩"，默认为"压缩"
            
        Returns:
            调整后的E值和深度值
        """
        # 获取通道角度（俯仰角）
        # 1. 先从channel_config获取默认角度
        pitch_angle = self.channel_config.get(channel, {}).get('angle', 0)
        # 2. 然后用提供的beam_angle替换默认角度（如果有）
        if beam_angle is not None:
            pitch_angle = beam_angle
        # 3. 最后根据beam_way值调整角度正负，beam_way值为-1表示后向通道，角度为负
        if beam_way == -1:
            pitch_angle = -abs(pitch_angle)
        
        # 定义水平角：斜向通道（0-3）为20度，其他通道为0度
        if 0 <= channel <= 3:
            horizontal_angle = 20
        else:
            horizontal_angle = 0
        
        # 将角度转换为弧度
        pitch_rad = math.radians(pitch_angle)
        horizontal_rad = math.radians(horizontal_angle)
        
        # 计算cos(俯仰角)和sin(俯仰角)
        cos_pitch = math.cos(pitch_rad)
        sin_pitch = math.sin(pitch_rad)
        
        # 计算cos(水平角)
        cos_horizontal = math.cos(horizontal_rad)
        
        # 计算调整后的深度
        # 原始深度 = 声程 * cos(水平角) * cos(俯仰角)
        raw_depth = depth * cos_horizontal * cos_pitch
        # 根据显示模式调整深度，处理二次反射
        adjusted_depth = self.calculate_adjusted_depth(channel, raw_depth, display_mode)
        
        # 计算调整后的E值
        # 编码偏差 = 声程 * cos(水平角) * sin(俯仰角)
        # 对于B1点：调整后的E值 = 原始E值 - beam_offset + 编码偏差 / 3
        # 这里默认使用B1点的计算公式，因为聚类主要基于B1点
        adjusted_e = e - beam_offset + (depth * cos_horizontal * sin_pitch) / 3
        
        return adjusted_e, adjusted_depth
    
    def batch_convert_coordinates(self, 
                                points: List[Dict], 
                                beam_offset_map: Dict[int, float],
                                beam_way_map: Dict[int, int] = None,
                                beam_angle_map: Dict[int, float] = None,
                                display_mode: str = "压缩") -> List[Dict]:
        """
        批量转换坐标
        
        Args:
            points: 点列表，每个点包含{'e', 'channel', 'depth'}键
            beam_offset_map: 通道到拼孔值的映射
            beam_way_map: 通道到射向的映射（可选）
            beam_angle_map: 通道到声束角度的映射（可选）
            display_mode: 显示模式，"V型"或"压缩"，默认为"压缩"
            
        Returns:
            带有调整后坐标的点列表，每个点包含{'e', 'channel', 'depth', 'adjusted_e', 'adjusted_depth'}键
        """
        if beam_way_map is None:
            beam_way_map = {}
        if beam_angle_map is None:
            beam_angle_map = {}
        
        converted_points = []
        for point in points:
            e = point['e']
            channel = point['channel']
            depth = point['depth']
            
            beam_offset = beam_offset_map.get(channel, 0)
            beam_way = beam_way_map.get(channel, 0)
            beam_angle = beam_angle_map.get(channel, None)
            
            adjusted_e, adjusted_depth = self.calculate_display_coordinates(
                e, channel, depth, beam_offset, beam_way, beam_angle, display_mode
            )
            
            converted_point = point.copy()
            converted_point['adjusted_e'] = adjusted_e
            converted_point['adjusted_depth'] = adjusted_depth
            converted_points.append(converted_point)
        
        return converted_points


# 创建全局实例，方便直接调用
converter = CoordinateConverter()