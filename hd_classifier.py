"""
超维计算分类器
基于OnlineHD实现超声波伤损识别
"""

import os
import torch
import torchhd
import numpy as np
from typing import List, Tuple, Optional, Dict
from config import config
from hd_encoder import HDCEncoder

class HDCClasifier:
    """超维计算分类器"""
    
    def __init__(self, n_classes: int = 20, dimensions: int = None, 
                 vsa_type: str = None, classifier_type: str = None):
        self.n_classes = n_classes
        self.dimensions = dimensions or config.get("model.dimensions", 10000)
        self.vsa_type = vsa_type or config.get("model.vsa_type", "MAP")
        self.classifier_type = classifier_type or config.get("model.classifier", "OnlineHD")
        
        # 创建编码器
        self.encoder = HDCEncoder(self.dimensions, self.vsa_type)
        
        # 创建分类器
        if self.classifier_type == "OnlineHD":
            self.model = torchhd.OnlineHD(
                n_features=self.dimensions,
                n_dimensions=self.dimensions,
                n_classes=self.n_classes,
                epochs=config.get("model.epochs", 120),
                lr=config.get("model.learning_rate", 0.035),
                vsa=self.vsa_type
            )
        elif self.classifier_type == "AdaptHD":
            self.model = torchhd.AdaptHD(
                n_features=self.dimensions,
                n_dimensions=self.dimensions,
                n_classes=self.n_classes,
                n_levels=100,
                min_level=-1,
                max_level=1,
                epochs=config.get("model.epochs", 120),
                lr=config.get("model.learning_rate", 0.035),
                vsa=self.vsa_type
            )
        elif self.classifier_type == "Vanilla":
            self.model = torchhd.Vanilla(
                n_features=self.dimensions,
                n_dimensions=self.dimensions,
                n_classes=self.n_classes,
                n_levels=100,
                min_level=-1,
                max_level=1,
                vsa=self.vsa_type
            )
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")
        
        # 类别名称映射
        self.class_names = {
            0: "未识别",
            1: "轨头外侧核伤",
            2: "轨头中心核伤",
            3: "轨头内侧核伤",
            4: "轨头内、外、中心核伤",
            5: "轨头内、中心核伤",
            6: "轨头外、中心核伤",
            7: "轨头水平分裂",
            8: "轨头、轨腰分裂",
            9: "轨腰水平裂",
            10: "轨底断裂",
            11: "螺孔裂纹",
            12: "接头",
            13: "螺孔",
            14: "导线孔",
            15: "焊缝",
            16: "鱼鳞伤",
            17: "轨底锈蚀",
            18: "轨底焊筋",
            19: "道岔"
        }
    
    def train(self, data: np.ndarray, labels: np.ndarray, 
              epochs: int = None, lr: float = None) -> Dict:
        """训练分类器
        
        Args:
            data: 训练数据，形状为 [n_samples, n_features]
            labels: 训练标签，形状为 [n_samples]
            epochs: 训练轮数
            lr: 学习率
        
        Returns:
            训练信息字典
        """
        # 编码数据为超维向量
        hv_data = self.encode_data(data)
        
        # 转换为torch数据加载器
        train_loader = self._create_dataloader(hv_data, labels)
        
        # 训练模型
        self.model.fit(train_loader)
        
        return {
            "n_samples": len(data),
            "n_classes": self.n_classes,
            "dimensions": self.dimensions
        }
    
    def predict(self, data: np.ndarray, threshold: float = None) -> np.ndarray:
        """预测数据
        
        Args:
            data: 测试数据，形状为 [n_samples, n_features]
            threshold: 相似度阈值
        
        Returns:
            预测标签，形状为 [n_samples]
        """
        # 编码数据为超维向量
        hv_data = self.encode_data(data)
        
        # 预测
        with torch.no_grad():
            logits = self.model(hv_data)
            predictions = torch.argmax(logits, dim=1)
        
        # 应用阈值过滤
        if threshold is not None:
            with torch.no_grad():
                similarities = torch.max(logits, dim=1)[0]
                below_threshold = similarities < threshold
                predictions[below_threshold] = 0  # 低于阈值标记为未识别
        
        return predictions.numpy()
    
    def predict_with_confidence(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测数据并返回置信度
        
        Args:
            data: 测试数据，形状为 [n_samples, n_features]
        
        Returns:
            predictions: 预测标签
            confidences: 置信度
        """
        # 编码数据为超维向量
        hv_data = self.encode_data(data)
        
        # 预测
        with torch.no_grad():
            logits = self.model(hv_data)
            predictions = torch.argmax(logits, dim=1)
            confidences = torch.max(logits, dim=1)[0]
        
        return predictions.numpy(), confidences.numpy()
    
    def encode_data(self, data: np.ndarray) -> torch.Tensor:
        """编码数据为超维向量
        
        Args:
            data: 输入数据，形状为 [n_samples, n_features]
        
        Returns:
            超维向量，形状为 [n_samples, dimensions]
        """
        n_samples = data.shape[0]
        hv_list = []
        
        for i in range(n_samples):
            # 使用多通道编码
            hv = self.encoder.encode_multi_channel_sample(data[i])
            hv_list.append(hv)
        
        return torch.stack(hv_list, dim=0)
    
    def _create_dataloader(self, data: torch.Tensor, labels: np.ndarray, 
                           batch_size: int = 32):
        """创建数据加载器"""
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(data, torch.from_numpy(labels).long())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """评估分类器性能
        
        Args:
            data: 测试数据
            labels: 真实标签
        
        Returns:
            评估结果字典
        """
        predictions = self.predict(data)
        
        # 计算准确率
        accuracy = np.mean(predictions == labels)
        
        # 计算每个类别的精确率和召回率
        precision_per_class = {}
        recall_per_class = {}
        
        for class_id in range(self.n_classes):
            # 精确率
            mask = predictions == class_id
            if mask.sum() > 0:
                precision = (predictions[mask] == labels[mask]).sum() / mask.sum()
            else:
                precision = 0.0
            precision_per_class[class_id] = precision
            
            # 召回率
            mask = labels == class_id
            if mask.sum() > 0:
                recall = (predictions[mask] == labels[mask]).sum() / mask.sum()
            else:
                recall = 0.0
            recall_per_class[class_id] = recall
        
        return {
            "accuracy": accuracy,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class
        }
    
    def save_model(self, filepath: str) -> None:
        """保存模型
        
        Args:
            filepath: 模型保存路径
        """
        model_state = {
            "model_state_dict": self.model.state_dict(),
            "dimensions": self.dimensions,
            "vsa_type": self.vsa_type,
            "classifier_type": self.classifier_type,
            "n_classes": self.n_classes
        }
        
        torch.save(model_state, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载模型
        
        Args:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        model_state = torch.load(filepath)
        
        self.dimensions = model_state["dimensions"]
        self.vsa_type = model_state["vsa_type"]
        self.classifier_type = model_state["classifier_type"]
        self.n_classes = model_state["n_classes"]
        
        # 重新创建模型
        if self.classifier_type == "OnlineHD":
            self.model = torchhd.OnlineHD(
                n_features=self.dimensions,
                n_dimensions=self.dimensions,
                n_classes=self.n_classes,
                vsa=self.vsa_type
            )
        elif self.classifier_type == "AdaptHD":
            self.model = torchhd.AdaptHD(
                n_features=self.dimensions,
                n_dimensions=self.dimensions,
                n_classes=self.n_classes,
                vsa=self.vsa_type
            )
        elif self.classifier_type == "Vanilla":
            self.model = torchhd.Vanilla(
                n_features=self.dimensions,
                n_dimensions=self.dimensions,
                n_classes=self.n_classes,
                vsa=self.vsa_type
            )
        
        # 加载权重
        self.model.load_state_dict(model_state["model_state_dict"])
        print(f"模型已从 {filepath} 加载")
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称
        
        Args:
            class_id: 类别ID
        
        Returns:
            类别名称
        """
        return self.class_names.get(class_id, f"未知类别_{class_id}")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "n_classes": self.n_classes,
            "dimensions": self.dimensions,
            "vsa_type": self.vsa_type,
            "classifier_type": self.classifier_type,
            "class_names": self.class_names
        }