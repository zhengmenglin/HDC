"""
实时处理器
实现数据采集、实时识别
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
from config import config
from hd_classifier import HDCClasifier

class RealtimeProcessor:
    """实时处理器"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(
            config.get("data.output_dir", "output"), 
            "hd_model.pth"
        )
        
        # 加载模型
        self.classifier = HDCClasifier()
        if os.path.exists(self.model_path):
            self.classifier.load_model(self.model_path)
            print(f"模型已加载: {self.model_path}")
        else:
            print(f"警告: 模型文件不存在: {self.model_path}")
        
        # 实时参数
        self.threshold = config.get("realtime.threshold", 0.5)
        self.update_interval = config.get("realtime.update_interval", 100)
        self.batch_size = config.get("realtime.batch_size", 1)
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=1000)
        self.prediction_buffer = deque(maxlen=1000)
        
        # 统计信息
        self.total_samples = 0
        self.correct_predictions = 0
        self.false_negatives = 0  # 漏检数
        self.false_positives = 0  # 误报数
        
        # 类别计数
        self.class_counts = {}
    
    def process_sample(self, sample: np.ndarray, 
                      true_label: int = None) -> Dict:
        """处理单个样本
        
        Args:
            sample: 输入样本，形状为 [10] (编码值 + 9通道深度值)
            true_label: 真实标签（可选，用于评估）
        
        Returns:
            处理结果字典
        """
        # 预测
        predictions, confidences = self.classifier.predict_with_confidence(
            sample.reshape(1, -1)
        )
        
        predicted_class = int(predictions[0])
        confidence = float(confidences[0])
        
        # 应用阈值
        if confidence < self.threshold:
            predicted_class = 0  # 标记为未识别
            confidence = 0.0
        
        # 更新统计
        self.total_samples += 1
        self.class_counts[predicted_class] = self.class_counts.get(predicted_class, 0) + 1
        
        # 如果有真实标签，更新准确率
        if true_label is not None:
            if predicted_class == true_label:
                self.correct_predictions += 1
            elif true_label != 0 and predicted_class == 0:
                # 真实是伤损，但预测为未识别 -> 漏检
                self.false_negatives += 1
            elif true_label == 0 and predicted_class != 0:
                # 真实是正常，但预测为伤损 -> 误报
                self.false_positives += 1
        
        # 添加到缓冲区
        result = {
            "sample": sample.tolist(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "true_label": true_label,
            "class_name": self.classifier.get_class_name(predicted_class),
            "timestamp": self.total_samples
        }
        
        self.data_buffer.append(result)
        self.prediction_buffer.append(predicted_class)
        
        return result
    
    def process_batch(self, samples: np.ndarray, 
                     true_labels: np.ndarray = None) -> List[Dict]:
        """批量处理样本
        
        Args:
            samples: 输入样本，形状为 [n_samples, 10]
            true_labels: 真实标签（可选）
        
        Returns:
            处理结果列表
        """
        results = []
        
        for i in range(len(samples)):
            sample = samples[i]
            true_label = true_labels[i] if true_labels is not None else None
            result = self.process_sample(sample, true_label)
            results.append(result)
        
        return results
    
    def process_stream(self, data_stream: List[np.ndarray], 
                      true_labels: List[int] = None) -> List[Dict]:
        """处理数据流
        
        Args:
            data_stream: 数据流
            true_labels: 真实标签列表（可选）
        
        Returns:
            所有处理结果
        """
        all_results = []
        
        for i, sample in enumerate(data_stream):
            true_label = true_labels[i] if true_labels is not None else None
            result = self.process_sample(sample, true_label)
            all_results.append(result)
            
            # 定期更新模型（增量学习）
            if self.total_samples % self.update_interval == 0:
                self.update_model_incremental()
        
        return all_results
    
    def update_model_incremental(self, new_data: np.ndarray = None,
                               new_labels: np.ndarray = None) -> None:
        """增量更新模型
        
        Args:
            new_data: 新数据
            new_labels: 新标签
        """
        if new_data is None or len(self.data_buffer) == 0:
            return
        
        # 使用缓冲区中的数据进行增量训练
        buffer_data = np.array([r["sample"] for r in self.data_buffer])
        buffer_labels = np.array([r["true_label"] for r in self.data_buffer 
                                if r["true_label"] is not None])
        
        if len(buffer_labels) > 0:
            try:
                self.classifier.train(buffer_data, buffer_labels, lr=0.01)
                print(f"模型已增量更新，样本数: {len(buffer_labels)}")
            except Exception as e:
                print(f"增量更新失败: {e}")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        accuracy = self.correct_predictions / self.total_samples if self.total_samples > 0 else 0.0
        fnr = self.false_negatives / self.total_samples if self.total_samples > 0 else 0.0
        fpr = self.false_positives / self.total_samples if self.total_samples > 0 else 0.0
        
        return {
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions,
            "accuracy": accuracy,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "false_negative_rate": fnr,
            "false_positive_rate": fpr,
            "class_counts": self.class_counts
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.total_samples = 0
        self.correct_predictions = 0
        self.false_negatives = 0
        self.false_positives = 0
        self.class_counts = {}
        self.data_buffer.clear()
        self.prediction_buffer.clear()
        print("统计信息已重置")
    
    def save_predictions(self, filepath: str) -> None:
        """保存预测结果"""
        results = list(self.prediction_buffer)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"预测结果已保存到: {filepath}")
    
    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """获取最近的预测结果"""
        recent_results = list(self.data_buffer)[-n:]
        return recent_results
    
    def set_threshold(self, threshold: float) -> None:
        """设置相似度阈值"""
        self.threshold = threshold
        print(f"相似度阈值已设置为: {threshold}")
    
    def set_update_interval(self, interval: int) -> None:
        """设置模型更新间隔"""
        self.update_interval = interval
        print(f"模型更新间隔已设置为: {interval}")
    
    def generate_annotation_json(self, predictions: List[Dict], 
                              output_path: str) -> None:
        """生成标注JSON文件
        
        Args:
            predictions: 预测结果列表
            output_path: 输出文件路径
        """
        annotations = []
        
        for i, pred in enumerate(predictions):
            if pred["predicted_class"] != 0:  # 只标注识别为伤损的点
                annotation = {
                    "id": str(i),
                    "type": "伤损",
                    "subtype": {
                        "id": str(pred["predicted_class"]),
                        "name": pred["class_name"]
                    },
                    "points": [{
                        "e": pred.get("timestamp", i),
                        "channel": int(pred["sample"][1]),  # 取通道号
                        "depth": pred["sample"][2],  # 取深度值
                        "amplitude": 100.0  # 默认幅度
                    }]
                }
                annotations.append(annotation)
        
        output_data = {"annotations": annotations}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"标注文件已生成: {output_path}")
    
    def process_json_data(self, json_data: dict) -> List[Dict]:
        """处理JSON格式的数据
        
        Args:
            json_data: JSON格式的数据
        
        Returns:
            处理结果列表
        """
        samples = []
        
        # 解析B扫描数据
        for key, value in json_data.items():
            if key.startswith("B"):
                try:
                    bscan_data = value
                    if isinstance(bscan_data, list) and len(bscan_data) >= 2:
                        channel = int(bscan_data[0])
                        depth_values = bscan_data[1:]
                        
                        # 构建样本: [编码值, 9通道深度值]
                        encoding_value = int(key[1:])
                        sample = [encoding_value] + [0.0] * 9
                        sample[channel + 1] = depth_values[0] if depth_values else 0.0
                        
                        samples.append(sample)
                except (ValueError, IndexError, TypeError):
                    continue
        
        # 批量处理
        if samples:
            samples_array = np.array(samples)
            results = self.process_batch(samples_array)
            return results
        
        return []