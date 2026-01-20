"""
训练器
实现数据加载、模型训练、保存
"""

import os
import json
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from config import config
from ultrasonic_data_converter import UltrasonicDataConverter
from hd_classifier import HDCClasifier
from evaluator import Evaluator

class Trainer:
    """模型训练器"""
    
    def __init__(self):
        self.converter = UltrasonicDataConverter()
        self.evaluator = Evaluator()
    
    def prepare_data(self, input_dir: str = None, 
                   output_dir: str = None,
                   normalize: bool = True,
                   window_sampling: bool = False,
                   window_size: int = 64,
                   train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        """准备训练数据"""
        input_dir = input_dir or config.get("data.input_dir", "data")
        output_dir = output_dir or config.get("data.output_dir", "output")
        
        # 转换数据格式
        csv_files = self.converter.convert_directory(input_dir, output_dir)
        
        # 加载并合并所有数据
        all_data = []
        all_labels = []
        
        for csv_file in csv_files:
            data, labels = self.converter.load_csv(csv_file)
            all_data.append(data)
            all_labels.append(labels)
        
        # 合并数据
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        # 数据归一化
        if normalize:
            combined_data = self.converter.normalize_data(combined_data)
        
        # 滑动窗口采样
        if window_sampling:
            combined_data, combined_labels = self.converter.apply_window_sampling(
                combined_data, combined_labels, window_size
            )
        
        # 划分训练集和测试集
        train_data, train_labels, test_data, test_labels = self.converter.split_train_test(
                combined_data, combined_labels, train_ratio
            )
        
        return train_data, train_labels, test_data, test_labels
    
    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray, 
                  test_data: np.ndarray = None, test_labels: np.ndarray = None,
                  model_path: str = None, save_path: str = None) -> Dict:
        """训练模型"""
        output_dir = config.get("data.output_dir", "output")
        
        # 创建分类器
        classifier = HDCClasifier()
        
        # 训练模型
        train_info = classifier.train(train_data, train_labels)
        
        # 保存模型
        if model_path is None:
            model_path = os.path.join(output_dir, "hd_model.pth")
        if save_path is None:
            save_path = os.path.join(output_dir, "hd_model_info.json")
        
        classifier.save_model(model_path)
        
        # 保存训练信息
        if save_path is not None:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(train_info, f, indent=2)
        
        # 评估模型
        if test_data is not None and test_labels is not None:
            eval_results = self.evaluator.evaluate(classifier, test_data, test_labels)
            
            # 保存评估结果
            eval_path = os.path.join(output_dir, "evaluation_results.json")
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2)
            
            print(f"评估结果已保存到: {eval_path}")
        
        return {
            "model_path": model_path,
            "train_info": train_info,
            "eval_results": eval_results if test_data is not None else None
        }
    
    def train_from_files(self, train_csv: str, test_csv: str = None, 
                      model_path: str = None, save_path: str = None) -> Dict:
        """从CSV文件训练模型"""
        # 加载数据
        train_data, train_labels = self.converter.load_csv(train_csv)
        
        # 加载测试数据（可选）
        test_data = None
        test_labels = None
        if test_csv is not None:
            test_data, test_labels = self.converter.load_csv(test_csv)
        
        # 训练模型
        results = self.train_model(train_data, train_labels, test_data, test_labels, model_path, save_path)
        
        return results
    
    def cross_validate(self, data: np.ndarray, labels: np.ndarray, 
                   folds: int = 5) -> Dict:
        """交叉验证"""
        n_samples = len(data)
        fold_size = n_samples // folds
        
        fold_results = []
        
        for fold in range(folds):
            # 划分训练集和验证集
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < folds - 1 else n_samples
            
            test_indices = list(range(start_idx, end_idx))
            train_indices = list(set(range(n_samples)) - set(test_indices))
            
            train_data = data[train_indices]
            train_labels = labels[train_indices]
            test_data = data[test_indices]
            test_labels = labels[test_indices]
            
            # 训练模型
            results = self.train_model(train_data, train_labels, test_data, test_labels)
            fold_results.append(results)
        
        # 计算平均结果
        avg_results = {}
        if fold_results:
            # 合并所有折的训练信息
            total_samples = sum(result["train_info"]["n_samples"] for result in fold_results)
            
            # 合并评估结果
            all_eval_results = []
            for result in fold_results:
                if result["eval_results"] is not None:
                    all_eval_results.append(result["eval_results"])
            
            # 计算平均指标
            if all_eval_results:
                avg_accuracy = np.mean([r["accuracy"] for r in all_eval_results])
                avg_precision = {}
                avg_recall = {}
                
                for class_id in range(fold_results[0]["train_info"]["n_classes"]):
                    class_precisions = [r["precision_per_class"][class_id] for r in all_eval_results]
                    class_recalls = [r["recall_per_class"][class_id] for r in all_eval_results]
                    
                    avg_precision[class_id] = np.mean(class_precisions)
                    avg_recall[class_id] = np.mean(class_recalls)
                
                avg_results = {
                    "accuracy": avg_accuracy,
                    "precision_per_class": avg_precision,
                    "recall_per_class": avg_recall
                }
        
        return {
            "fold_results": fold_results,
            "average_results": avg_results,
            "total_samples": total_samples
        }
    
    def train_incremental(self, model_path: str, new_data: np.ndarray, 
                      new_labels: np.ndarray, lr: float = 0.035) -> Dict:
        """增量训练模型"""
        # 加载现有模型
        classifier = HDCClasifier()
        classifier.load_model(model_path)
        
        # 增量训练
        train_info = classifier.train(new_data, new_labels, lr=lr)
        
        return {
            "model_path": model_path,
            "train_info": train_info
        }