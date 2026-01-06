import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DataPoint:
    """数据点类，存储从文件读取的数据"""
    channel: int
    encoding: int
    depth: float
    db_value: float
    amplitude: float = 100.0  # 信号幅度值（%），默认为100%

class HexDataReader:
    """16进制数据文件读取器"""
    
    def __init__(self, protocol_version: str = "1.0"):
        """
        初始化数据读取器
        
        Args:
            protocol_version: 协议版本
        """
        self.protocol_version = protocol_version
        
    def parse_hex_float(self, hex_str: str) -> float:
        """
        将16进制字符串转换为浮点数
        
        Args:
            hex_str: 16进制字符串
            
        Returns:
            转换后的浮点数
        """
        # 移除可能的前缀
        hex_str = hex_str.replace("0x", "").replace("0X", "")
        # 转换为整数，然后转换为浮点数
        return float(np.frombuffer(bytes.fromhex(hex_str), dtype=np.float32)[0])
    
    def read_data_file(self, file_path: str) -> List[DataPoint]:
        """
        读取数据文件，支持十六进制格式和JSON格式
        
        十六进制格式：每行包含4个16进制字段，用空格分隔
        字段顺序：通道号 编码值 深度值(dB值) 原始dB值
        
        JSON格式：每行一个JSON对象，格式为 {"Bn": [通道号, 深度值, 幅度值]}
        例如：{"B1": [1, 125, 80]} 表示通道1，深度125mm，幅度80%
        
        Args:
            file_path: 数据文件路径

        Returns:
            读取的数据点列表
        """
        data_points = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 检测文件格式
            is_json = False
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    is_json = line.startswith('{') and line.endswith('}')
                    break
            
            # 解析文件内容
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释
                    continue
                
                if is_json:
                    # JSON格式解析
                    try:
                        # 解析JSON
                        json_data = json.loads(line)
                        
                        # 遍历所有B显点
                        for b_point_id, b_point_data in json_data.items():
                            if isinstance(b_point_data, list) and len(b_point_data) >= 3:
                                # 提取数据
                                channel = b_point_data[0]  # 通道号
                                encoding = int(b_point_id[1:])  # 从B1、B2等中提取编码值
                                depth = b_point_data[1]  # 深度值（mm）
                                amplitude = b_point_data[2]  # 幅度值（%）
                                
                                # 计算dB值（这里假设dB值与幅度值线性相关，可根据实际情况调整）
                                # 幅度值80%对应dB值约40dB，100%对应约50dB
                                db_value = 20 * np.log10(amplitude / 100) + 50
                                
                                # 创建数据点
                                data_point = DataPoint(channel, encoding, depth, db_value, amplitude)
                                data_points.append(data_point)
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行失败: {line}, 错误: {e}")
                        continue
                    except Exception as e:
                        print(f"处理JSON数据失败: {line}, 错误: {e}")
                        continue
                else:
                    # 十六进制格式解析
                    # 按空格分割字段
                    fields = line.split()
                    if len(fields) < 4:
                        continue  # 跳过格式不正确的行
                    
                    # 解析字段
                    try:
                        channel = int(fields[0], 16)  # 通道号（16进制）
                        encoding = int(fields[1], 16)  # 编码值（16进制）
                        depth_hex = fields[2]  # 深度值（16进制浮点数）
                        db_hex = fields[3]  # dB值（16进制浮点数）
                        
                        # 转换16进制浮点数
                        depth = self.parse_hex_float(depth_hex)
                        db_value = self.parse_hex_float(db_hex)
                        
                        # 计算幅度值（假设dB值与幅度值线性相关）
                        # 幅度值80%对应dB值约40dB，100%对应约50dB
                        amplitude = 100 * 10 ** ((db_value - 50) / 20)
                        amplitude = max(0, min(100, amplitude))  # 限制在0-100%之间
                        
                        # 创建数据点，包含幅度值
                        data_point = DataPoint(channel, encoding, depth, db_value, amplitude)
                        data_points.append(data_point)
                    except Exception as e:
                        print(f"解析行失败: {line}, 错误: {e}")
                        continue
                        
        except Exception as e:
            print(f"读取文件失败: {e}")
        
        return data_points
    
    def convert_to_training_data(self, data_points: List[DataPoint], labels_file: str) -> Dict[str, List[Dict[str, float]]]:
        """
        将数据点转换为训练数据格式
        
        Args:
            data_points: 数据点列表
            labels_file: 标签文件路径
            
        Returns:
            训练数据字典
        """
        training_data = {}
        
        # 读取标签文件
        labels = {}
        try:
            with open(labels_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    fields = line.split()
                    if len(fields) < 3:
                        continue
                    # 支持十六进制和十进制格式的通道号和编码值
                    try:
                        channel = int(fields[0], 16)
                        encoding = int(fields[1], 16)
                    except ValueError:
                        # 如果是十进制格式
                        channel = int(fields[0])
                        encoding = int(fields[1])
                    label = fields[2]
                    labels[(channel, encoding)] = label
        except Exception as e:
            print(f"读取标签文件失败: {e}")
            return training_data
        
        # 转换数据点为训练数据
        for data_point in data_points:
            key = (data_point.channel, data_point.encoding)
            if key in labels:
                label = labels[key]
                if label not in training_data:
                    training_data[label] = []
                
                # 创建训练样本，包含幅度值
                sample = {
                    'depth': data_point.depth,
                    'db_value': data_point.db_value,
                    'amplitude': data_point.amplitude,  # 新增幅度值字段
                    'binary_value': 1.0 if data_point.db_value > 30.0 else 0.0,
                    'channel': float(data_point.channel),
                    'encoding': float(data_point.encoding)
                }
                training_data[label].append(sample)
        
        return training_data
    
    def convert_to_test_data(self, data_points: List[DataPoint]) -> List[Dict[str, float]]:
        """
        将数据点转换为测试数据格式
        
        Args:
            data_points: 数据点列表
            
        Returns:
            测试数据列表
        """
        test_data = []
        
        for data_point in data_points:
            # 创建测试样本，包含幅度值
            sample = {
                'depth': data_point.depth,
                'db_value': data_point.db_value,
                'amplitude': data_point.amplitude,  # 新增幅度值字段
                'binary_value': 1.0 if data_point.db_value > 30.0 else 0.0,
                'channel': float(data_point.channel),
                'encoding': float(data_point.encoding)
            }
            test_data.append(sample)
        
        return test_data

# 示例用法
if __name__ == "__main__":
    # 创建数据读取器
    reader = HexDataReader()
    
    # 示例：读取训练数据
    # train_data_points = reader.read_data_file('train_data.hex')
    # train_data = reader.convert_to_training_data(train_data_points, 'train_labels.txt')
    # print(f"读取到 {len(train_data)} 类训练数据")
    # 
    # # 示例：读取测试数据
    # test_data_points = reader.read_data_file('test_data.hex')
    # test_data = reader.convert_to_test_data(test_data_points)
    # print(f"读取到 {len(test_data)} 个测试样本")