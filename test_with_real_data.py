"""
使用实际数据测试超声波探伤HDC系统
"""

import os
import numpy as np
from data_adapter import DataAdapter
from config import config
from trainer import Trainer
from evaluator import Evaluator
from hd_classifier import HDCClasifier
from main_controller import MainController

def test_with_real_data():
    """使用实际数据测试系统"""
    print("="*60)
    print("超声波探伤HDC系统 - 实际数据测试")
    print("="*60)
    
    # 创建数据适配器
    adapter = DataAdapter()
    
    # 1. 转换训练数据
    print("\n步骤1: 转换训练数据")
    train_dir = config.get("data.train_dir", "data/train")
    train_files = []
    
    for filename in os.listdir(train_dir):
        if filename.endswith('.json') and not filename.endswith('标注.json'):
            json_file = os.path.join(train_dir, filename)
            base_name = os.path.splitext(filename)[0]
            annotation_file = os.path.join(train_dir, f'{base_name}标注.json')
            
            # 转换文件
            output_csv = adapter.convert_to_csv(
                adapter.load_original_json(json_file),
                annotation_file if os.path.exists(annotation_file) else None,
                os.path.join(config.get("data.output_dir", "output"), f'{base_name}_converted.csv')
            )
            train_files.append(output_csv)
    
    print(f"已转换 {len(train_files)} 个训练文件")
    
    # 2. 转换测试数据
    print("\n步骤2: 转换测试数据")
    test_dir = config.get("data.test_dir", "data/test")
    test_files = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.json') and not filename.endswith('标注.json'):
            json_file = os.path.join(test_dir, filename)
            base_name = os.path.splitext(filename)[0]
            annotation_file = os.path.join(test_dir, f'{base_name}标注.json')
            
            # 转换文件
            output_csv = adapter.convert_to_csv(
                adapter.load_original_json(json_file),
                annotation_file if os.path.exists(annotation_file) else None,
                os.path.join(config.get("data.output_dir", "output"), f'{base_name}_converted.csv')
            )
            test_files.append(output_csv)
    
    print(f"已转换 {len(test_files)} 个测试文件")
    
    # 3. 合并训练数据
    print("\n步骤3: 合并训练数据")
    output_dir = config.get("data.output_dir", "output")
    train_csv = os.path.join(output_dir, "train_merged.csv")
    adapter.merge_csv_files(train_files, train_csv)
    
    # 4. 合并测试数据
    print("\n步骤4: 合并测试数据")
    test_csv = os.path.join(output_dir, "test_merged.csv")
    adapter.merge_csv_files(test_files, test_csv)
    
    # 5. 训练模型
    print("\n步骤5: 训练HDC模型")
    controller = MainController()
    
    # 加载训练数据
    train_data, train_labels = adapter.load_csv(train_csv)
    
    # 加载测试数据
    test_data, test_labels = adapter.load_csv(test_csv)
    
    # 训练模型
    train_results = controller.train_model(
        train_csv=train_csv,
        test_csv=test_csv,
        model_path=os.path.join(output_dir, "hd_model_real_data.pth"),
        save_path=os.path.join(output_dir, "hd_model_info.json")
    )
    
    print(f"训练完成，模型保存到: {train_results['model_path']}")
    
    # 6. 评估模型
    print("\n步骤6: 评估模型性能")
    eval_results = controller.evaluate_model(
        test_csv=test_csv,
        model_path=train_results['model_path']
    )
    
    # 7. 生成预测结果
    print("\n步骤7: 生成预测结果")
    
    # 加载训练好的模型
    classifier = HDCClasifier()
    classifier.load_model(train_results['model_path'])
    
    # 预测测试数据
    predictions = classifier.predict(test_data)
    
    # 生成预测标注文件
    output_annotations = []
    for i, (data_row, pred_label, true_label) in enumerate(zip(test_data, predictions, test_labels)):
        if pred_label != 0:  # 只标注识别为伤损的点
            # 找到第一个非零深度值的通道
            channel_idx = -1
            for j in range(1, 10):
                if data_row[j] > 0:
                    channel_idx = j - 1
                    break
            
            if channel_idx >= 0:
                annotation = {
                    "id": str(i),
                    "type": "伤损" if pred_label <= 11 else "结构",
                    "subtype": {
                        "id": str(pred_label),
                        "name": classifier.get_class_name(pred_label)
                    },
                    "points": [{
                        "e": int(data_row[0]),
                        "channel": channel_idx,
                        "depth": float(data_row[channel_idx + 1]),
                        "amplitude": 100.0
                    }]
                }
                output_annotations.append(annotation)
    
    # 保存预测结果
    pred_annotation_file = os.path.join(output_dir, "test_predictions.json")
    with open(pred_annotation_file, 'w', encoding='utf-8') as f:
        import json
        json.dump({"annotations": output_annotations}, f, indent=2, ensure_ascii=False)
    
    print(f"预测结果已保存到: {pred_annotation_file}")
    
    # 8. 统计信息
    print("\n步骤8: 统计信息")
    print(f"训练样本数: {len(train_data)}")
    print(f"测试样本数: {len(test_data)}")
    print(f"预测为伤损的样本数: {np.sum(predictions != 0)}")
    print(f"实际为伤损的样本数: {np.sum(test_labels != 0)}")
    
    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    print(f"总体准确率: {accuracy:.4f}")
    
    # 计算伤损类别的准确率
    damage_mask = test_labels != 0
    if damage_mask.sum() > 0:
        damage_accuracy = np.mean(predictions[damage_mask] == test_labels[damage_mask])
        print(f"伤损类别准确率: {damage_accuracy:.4f}")
    
    # 计算正常类别的准确率
    normal_mask = test_labels == 0
    if normal_mask.sum() > 0:
        normal_accuracy = np.mean(predictions[normal_mask] == test_labels[normal_mask])
        print(f"正常类别准确率: {normal_accuracy:.4f}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    return {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "model_path": train_results['model_path'],
        "pred_annotation_file": pred_annotation_file,
        "accuracy": accuracy,
        "eval_results": eval_results
    }

def quick_test():
    """快速测试 - 只使用部分数据"""
    print("="*60)
    print("超声波探伤HDC系统 - 快速测试")
    print("="*60)
    
    # 创建数据适配器
    adapter = DataAdapter()
    
    # 只使用一个训练文件和一个测试文件
    train_dir = config.get("data.train_dir", "data/train")
    test_dir = config.get("data.test_dir", "data/test")
    
    # 选择第一个训练文件
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.json') and not f.endswith('标注.json')]
    if not train_files:
        print("未找到训练文件")
        return
    
    train_file = os.path.join(train_dir, train_files[0])
    train_base = os.path.splitext(train_files[0])[0]
    train_annotation = os.path.join(train_dir, f'{train_base}标注.json')
    
    # 选择第一个测试文件
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.json') and not f.endswith('标注.json')]
    if not test_files:
        print("未找到测试文件")
        return
    
    test_file = os.path.join(test_dir, test_files[0])
    test_base = os.path.splitext(test_files[0])[0]
    test_annotation = os.path.join(test_dir, f'{test_base}标注.json')
    
    # 转换文件
    output_dir = config.get("data.output_dir", "output")
    train_csv = adapter.convert_to_csv(
        adapter.load_original_json(train_file),
        train_annotation if os.path.exists(train_annotation) else None,
        os.path.join(output_dir, f'{train_base}_converted.csv')
    )
    
    test_csv = adapter.convert_to_csv(
        adapter.load_original_json(test_file),
        test_annotation if os.path.exists(test_annotation) else None,
        os.path.join(output_dir, f'{test_base}_converted.csv')
    )
    
    print(f"已转换训练文件: {train_csv}")
    print(f"已转换测试文件: {test_csv}")
    
    # 训练模型
    controller = MainController()
    train_results = controller.train_model(
        train_csv=train_csv,
        test_csv=test_csv,
        model_path=os.path.join(output_dir, "hd_model_quick.pth"),
        save_path=os.path.join(output_dir, "hd_model_quick_info.json")
    )
    
    # 评估模型
    eval_results = controller.evaluate_model(
        test_csv=test_csv,
        model_path=train_results['model_path']
    )
    
    print(f"快速测试完成，准确率: {eval_results.get('accuracy', 0):.4f}")
    
    return train_results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        test_with_real_data()