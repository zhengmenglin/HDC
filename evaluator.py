"""
评估器
实现多指标评估：F1分数、MCC、AUC、混淆矩阵等
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize
from config import config
from hd_classifier import HDCClasifier

class Evaluator:
    """模型评估器"""
    
    def __init__(self):
        self.metrics = config.get("evaluation.metrics", 
                                ["accuracy", "precision", "recall", "f1_score", "mcc", "auc"])
    
    def evaluate(self, classifier: HDCClasifier, data: np.ndarray, 
               labels: np.ndarray) -> Dict:
        """评估分类器性能
        
        Args:
            classifier: 训练好的分类器
            data: 测试数据
            labels: 真实标签
        
        Returns:
            评估结果字典
        """
        # 预测
        predictions = classifier.predict(data)
        confidences = None
        
        if hasattr(classifier, 'predict_with_confidence'):
            predictions, confidences = classifier.predict_with_confidence(data)
        
        # 计算各项指标
        results = {}
        
        # 准确率
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(labels, predictions)
        
        # 精确率（宏平均）
        if "precision" in self.metrics:
            results["precision_macro"] = precision_score(labels, predictions, average='macro', zero_division=0)
            results["precision_weighted"] = precision_score(labels, predictions, average='weighted', zero_division=0)
        
        # 召回率（宏平均）
        if "recall" in self.metrics:
            results["recall_macro"] = recall_score(labels, predictions, average='macro', zero_division=0)
            results["recall_weighted"] = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        # F1分数
        if "f1_score" in self.metrics:
            results["f1_score_macro"] = f1_score(labels, predictions, average='macro', zero_division=0)
            results["f1_score_weighted"] = f1_score(labels, predictions, average='weighted', zero_division=0)
            results["f1_score_samples"] = f1_score(labels, predictions, average='samples', zero_division=0)
        
        # MCC（马修斯相关系数）
        if "mcc" in self.metrics:
            results["mcc"] = self._calculate_mcc(labels, predictions, num_classes=classifier.n_classes)
        
        # AUC
        if "auc" in self.metrics:
            auc_scores = self._calculate_auc(labels, predictions, num_classes=classifier.n_classes)
            results["auc_macro"] = auc_scores["macro"]
            results["auc_weighted"] = auc_scores["weighted"]
            results["auc_per_class"] = auc_scores["per_class"]
        
        # 混淆矩阵
        results["confusion_matrix"] = self._calculate_confusion_matrix(labels, predictions, classifier.n_classes)
        
        # 每个类别的详细指标
        results["per_class_metrics"] = self._calculate_per_class_metrics(labels, predictions, classifier.n_classes)
        
        # 漏检率和误报率
        results["false_negative_rate"] = self._calculate_fnr(labels, predictions)
        results["false_positive_rate"] = self._calculate_fpr(labels, predictions)
        
        return results
    
    def _calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      num_classes: int) -> float:
        """计算马修斯相关系数"""
        from sklearn.metrics import matthews_corrcoef
        
        if num_classes > 2:
            # 多分类MCC
            mcc_scores = []
            for i in range(num_classes):
                binary_true = (y_true == i).astype(int)
                binary_pred = (y_pred == i).astype(int)
                try:
                    mcc = matthews_corrcoef(binary_true, binary_pred)
                    mcc_scores.append(mcc)
                except:
                    mcc_scores.append(0.0)
            return np.mean(mcc_scores)
        else:
            return matthews_corrcoef(y_true, y_pred)
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray,
                      num_classes: int) -> Dict:
        """计算AUC分数"""
        if num_classes <= 2:
            # 二分类
            try:
                auc = roc_auc_score(y_true, y_pred)
                return {"macro": auc, "weighted": auc, "per_class": {0: 0.5, 1: auc}}
            except:
                return {"macro": 0.5, "weighted": 0.5, "per_class": {}}
        
        # 多分类
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        y_pred_bin = label_binarize(y_pred, classes=list(range(num_classes)))
        
        auc_per_class = {}
        for i in range(num_classes):
            if y_true_bin[:, i].sum() > 0:
                try:
                    auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
                    auc_per_class[i] = auc
                except:
                    auc_per_class[i] = 0.5
            else:
                auc_per_class[i] = 0.5
        
        # 计算宏平均和加权平均
        valid_aucs = [v for v in auc_per_class.values() if v > 0]
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.5
        
        # 加权平均（按类别样本数加权）
        weights = []
        for i in range(num_classes):
            weights.append((y_true == i).sum())
        weights = np.array(weights) / len(y_true)
        weighted_auc = sum(auc_per_class.get(i, 0.5) * weights[i] for i in range(num_classes))
        
        return {
            "macro": macro_auc,
            "weighted": weighted_auc,
            "per_class": auc_per_class
        }
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   num_classes: int) -> List[List[int]]:
        """计算混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        return cm.tolist()
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    num_classes: int) -> Dict:
        """计算每个类别的指标"""
        metrics = {}
        
        for class_id in range(num_classes):
            true_mask = y_true == class_id
            pred_mask = y_pred == class_id
            
            # 真阳性
            tp = ((y_true == class_id) & (y_pred == class_id)).sum()
            # 假阳性
            fp = ((y_true != class_id) & (y_pred == class_id)).sum()
            # 假阴性
            fn = ((y_true == class_id) & (y_pred != class_id)).sum()
            # 真阴性
            tn = ((y_true != class_id) & (y_pred != class_id)).sum()
            
            # 精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # 召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[class_id] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "support": int(true_mask.sum())
            }
        
        return metrics
    
    def _calculate_fnr(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算漏检率"""
        # 定义正类为非0类（伤损类）
        positive_mask = y_true != 0
        if positive_mask.sum() == 0:
            return {"overall": 0.0, "per_class": {}}
        
        fn = ((y_true != 0) & (y_pred == 0)).sum()
        total_positive = positive_mask.sum()
        
        fnr_overall = fn / total_positive if total_positive > 0 else 0.0
        
        # 每个类别的漏检率
        fnr_per_class = {}
        for class_id in np.unique(y_true):
            class_mask = y_true == class_id
            class_fn = ((y_true == class_id) & (y_pred == 0)).sum()
            class_total = class_mask.sum()
            fnr_per_class[int(class_id)] = class_fn / class_total if class_total > 0 else 0.0
        
        return {"overall": fnr_overall, "per_class": fnr_per_class}
    
    def _calculate_fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算误报率"""
        # 定义负类为0类（正常/未识别）
        negative_mask = y_true == 0
        if negative_mask.sum() == 0:
            return {"overall": 0.0, "per_class": {}}
        
        fp = ((y_true == 0) & (y_pred != 0)).sum()
        total_negative = negative_mask.sum()
        
        fpr_overall = fp / total_negative if total_negative > 0 else 0.0
        
        return {"overall": fpr_overall}
    
    def calculate_f2_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                          beta: float = 2.0) -> float:
        """计算F2分数（召回率权重更高）"""
        # 使用加权精确率和召回率
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if precision + recall == 0:
            return 0.0
        
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        
        return f2
    
    def print_evaluation_report(self, results: Dict, classifier: HDCClasifier = None):
        """打印评估报告"""
        print("\n" + "="*60)
        print("模型评估报告")
        print("="*60)
        
        # 总体指标
        print("\n【总体指标】")
        print(f"准确率 (Accuracy): {results.get('accuracy', 'N/A'):.4f}")
        print(f"精确率 (Macro): {results.get('precision_macro', 'N/A'):.4f}")
        print(f"召回率 (Macro): {results.get('recall_macro', 'N/A'):.4f}")
        print(f"F1分数 (Macro): {results.get('f1_score_macro', 'N/A'):.4f}")
        print(f"F2分数: {results.get('f2_score', 'N/A'):.4f}")
        print(f"MCC: {results.get('mcc', 'N/A'):.4f}")
        print(f"AUC (Macro): {results.get('auc_macro', 'N/A'):.4f}")
        
        # 漏检率和误报率
        fnr = results.get('false_negative_rate', {})
        fpr = results.get('false_positive_rate', {})
        print(f"\n【错误率】")
        print(f"漏检率 (FNR): {fnr.get('overall', 'N/A'):.4f}")
        print(f"误报率 (FPR): {fpr.get('overall', 'N/A'):.4f}")
        
        # 关键指标评估
        print(f"\n【关键指标评估】")
        print(f"✓ F2分数 ≥ 0.85: {'是' if results.get('f2_score', 0) >= 0.85 else '否'} ({results.get('f2_score', 0):.4f})")
        print(f"✓ MCC ≥ 0.85: {'是' if results.get('mcc', 0) >= 0.85 else '否'} ({results.get('mcc', 0):.4f})")
        print(f"✓ AUC ≥ 0.95: {'是' if results.get('auc_macro', 0) >= 0.95 else '否'} ({results.get('auc_macro', 0):.4f})")
        print(f"✓ 漏检率 ≤ 1%: {'是' if fnr.get('overall', 1) <= 0.01 else '否'} ({fnr.get('overall', 1):.4f})")
        
        # 混淆矩阵
        cm = results.get('confusion_matrix', [])
        if cm:
            print(f"\n【混淆矩阵】")
            print("预测 →")
            for i, row in enumerate(cm):
                print(f"实际 {i}: {row}")
        
        # 类别详细指标
        print(f"\n【各类别指标】")
        per_class = results.get('per_class_metrics', {})
        for class_id, metrics in per_class.items():
            class_name = classifier.get_class_name(class_id) if classifier else str(class_id)
            print(f"{class_name} (ID={class_id}):")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1_score']:.4f}")
            print(f"  支持数: {metrics['support']}")
        
        print("\n" + "="*60)
    
    def save_evaluation_results(self, results: Dict, filepath: str):
        """保存评估结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"评估结果已保存到: {filepath}")