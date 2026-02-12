"""
超声波探伤系统配置文件
"""

import os
import json
from typing import Dict, Any

class Config:
    """系统配置管理类"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 数据处理配置
            "data": {
                "input_dir": "data",  # 原始数据目录
                "train_dir": "data/train",  # 训练数据目录
                "test_dir": "data/test",  # 测试数据目录
                "output_dir": "output",  # 输出目录
                "train_ratio": 0.7,  # 训练集比例
                "window_size": 128,  # 滑动窗口大小
                "downsample": 5,  # 降采样因子
                "normalize": True  # 是否归一化
            },
            
            # HDC模型配置
            "model": {
                "dimensions": 10000,  # 超维空间维度
                "vsa_type": "MAP",  # VSA类型: MAP, HRR, FHRR等
                "classifier": "OnlineHD",  # 分类器类型
                "learning_method": "onlinehd",  # 学习方式: onlinehd, ridge
                "learning_rate": 0.05,  # 学习率
                "ridge_alpha": 1.0,  # 岭回归正则化参数
                "epochs": 40,  # 训练轮数
                "model_path": "models/hdc_point_level_9ch.pth",  # 模型文件路径
                "learned_files_path": "models/hdc_point_level_9ch.json"  # 已学习文件列表路径
            },
            
            # 评估配置
            "evaluation": {
                "metrics": ["precision", "recall", "f1_score"],  # 评估指标
                "cross_validation": False,  # 是否使用交叉验证
                "folds": 5,  # 交叉验证折数
            },
            
            # 实时处理配置
            "realtime": {
                "threshold": 0.5,  # 相似度阈值
                "update_interval": 100,  # 模型更新间隔(样本数)
                "batch_size": 1,  # 批处理大小
            },
            
            # 可视化配置
            "visualization": {
                "b_scan_width": 800,  # B扫描图像宽度
                "b_scan_height": 600,  # B扫描图像高度
                "point_size": 5,  # 标注点大小
                "color_scheme": "viridis"  # 颜色方案
            },
            
            # 测试流程配置
            "test": {
                "results_dir": "hdc_test_results",  # 测试结果保存目录
                "train_strategy": "all",  # 训练策略: damage_only, mixed, all
                "predict_threshold": 0.7,  # 预测时的阈值
                "predict_margin": 0.0  # 预测时的margin，不再使用与正常类的比较
            }
        }
    
    def save_config(self) -> None:
        """保存配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return
        
        config[keys[-1]] = value
        self.config = config
        self.save_config()

# 全局配置实例
config = Config()