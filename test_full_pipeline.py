#!/usr/bin/env python3
"""
测试完整的超声波HDC管道
包括数据转换、训练、预测和结果转换
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import shutil

# 添加项目路径
sys.path.append('.')

# 导入超声波HDC程序
from ultrasonic_hdc_emg_style import train_model, predict_model
from ultrasonic_data_converter import UltrasonicDataConverter

def test_full_pipeline():
    """测试完整管道"""
    print("=== 开始完整超声波HDC管道测试 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # 配置参数
    WINDOW_SIZE = 64
    STEP = 32
    DIMENSIONS = 10000
    TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
    TEST_DATA_DIR = Path(r"d:\HDC\data\test")
    OUTPUT_DIR = Path("pipeline_test_results")
    TEMP_DIR = Path("pipeline_temp")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    try:
        # 1. 数据转换：训练数据
        print("\n=== Step 1: 训练数据转换 ===")
        converter = UltrasonicDataConverter()
        
        # 获取所有训练数据文件
        train_files = []
        for ext in ['*.json', '*.xxc.json']:
            train_files.extend(TRAIN_DATA_DIR.glob(ext))
        
        # 转换训练数据
        train_csv_files = []
        for data_file in train_files:
            # 跳过标注文件
            if '标注' in data_file.name:
                continue
                
            # 查找对应的标注文件（可选）
            annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            
            # 强制重新转换所有文件
            output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            
            # 转换文件，即使没有标注文件也转换
            try:
                result_file = converter.convert_single_file(
                    str(data_file), annotation_path, str(output_csv)
                )
                train_csv_files.append(str(output_csv))
                if annotation_path:
                    print(f"✓ 转换完成: {data_file.name} (使用标注文件)")
                else:
                    print(f"✓ 转换完成: {data_file.name} (无标注文件，label列全为0)")
            except Exception as e:
                print(f"✗ 转换失败 {data_file.name}: {e}")
                continue
        
        if not train_csv_files:
            print("[FAIL] 没有成功转换的训练数据文件")
            return False
        
        print(f"\n训练数据转换结果: {len(train_csv_files)} 个文件成功转换或已存在")
        
        # 2. 数据转换：测试数据
        print("\n=== Step 2: 测试数据转换 ===")
        
        # 获取所有测试数据文件
        test_files = []
        for ext in ['*.json', '*.xxc.json']:
            test_files.extend(TEST_DATA_DIR.glob(ext))
        
        # 转换测试数据
        test_csv_files = []
        for data_file in test_files:
            # 跳过标注文件
            if '标注' in data_file.name:
                continue
                
            # 查找对应的标注文件（可选）
            annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
            annotation_path = str(annotation_file) if annotation_file.exists() else None
            
            # 强制重新转换所有文件
            output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            
            # 转换文件，即使没有标注文件也转换
            try:
                result_file = converter.convert_single_file(
                    str(data_file), annotation_path, str(output_csv)
                )
                test_csv_files.append(str(output_csv))
                if annotation_path:
                    print(f"✓ 转换完成: {data_file.name} (使用标注文件)")
                else:
                    print(f"✓ 转换完成: {data_file.name} (无标注文件，label列全为0)")
            except Exception as e:
                print(f"✗ 转换失败 {data_file.name}: {e}")
                continue
        
        if not test_csv_files:
            print("[FAIL] 没有成功转换的测试数据文件")
            return False
        
        print(f"\n测试数据转换结果: {len(test_csv_files)} 个文件成功转换或已存在")
        
        # 3. 模型训练
        print("\n=== Step 3: 模型训练 ===")
        print(f"使用 {TRAIN_DATA_DIR} 目录下的所有CSV文件进行训练")
        print(f"超向量维度: {DIMENSIONS}")
        
        try:
            # 使用新的训练函数
            train_success = train_model()
            
            if not train_success:
                print("[FAIL] 模型训练失败")
                return False
            
            print(f"✓ 模型训练成功")
        except Exception as e:
            print(f"[FAIL] 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 模型预测
        print("\n=== Step 4: 模型预测 ===")
        print(f"使用 {TEST_DATA_DIR} 目录下的所有CSV文件进行预测")
        
        try:
            # 使用新的预测函数
            predict_success = predict_model()
            
            if not predict_success:
                print("[FAIL] 模型预测失败")
                return False
            
            print(f"✓ 模型预测完成")
        except Exception as e:
            print(f"[FAIL] 模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. CSV到HDC JSON转换
        print("\n=== Step 5: CSV到HDC JSON转换 ===")
        try:
            # 导入CSV到HDC JSON转换器
            from csv_to_hdc_json import CsvToHdcJsonConverter
            
            converter = CsvToHdcJsonConverter()
            
            # 转换所有测试CSV文件
            print(f"将 {TEST_DATA_DIR} 目录下的CSV文件转换为hdc.json格式")
            converter.batch_convert(str(TEST_DATA_DIR), eps=5.0, min_samples=5)
            
            # 转换所有训练CSV文件
            print(f"将 {TRAIN_DATA_DIR} 目录下的CSV文件转换为hdc.json格式")
            converter.batch_convert(str(TRAIN_DATA_DIR), eps=5.0, min_samples=5)
            
            print(f"✓ CSV到HDC JSON转换完成")
        except Exception as e:
            print(f"[FAIL] CSV到HDC JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 6. 结果评估
        print("\n=== Step 6: 结果评估 ===")
        print("注意：当前版本的评估需要手动检查生成的_hdc.csv文件和hdc.json文件")
        print("预测结果已保存到每个测试文件对应的_hdc.csv文件中")
        print("HDC JSON文件已保存到每个测试文件对应的_hdc.json文件中")
        
        # 输出最终结果
        print("\n" + "="*50)
        print("=== 完整超声波HDC管道测试完成 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"预测结果已保存到 {TEST_DATA_DIR} 目录下的_hdc.csv文件中")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 确保临时目录被清理
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

def calculate_accuracy(predicted_file, original_file):
    """计算识别正确率
    
    正确率定义为：识别正确的点数 / 所有识别出的点数
    
    Args:
        predicted_file: 包含预测结果的CSV文件路径
        original_file: 原始CSV文件路径（包含真实标签）
    
    Returns:
        accuracy: 正确率，范围0-1
    """
    # 读取预测结果文件
    pred_df = pd.read_csv(predicted_file)
    # 读取原始文件
    orig_df = pd.read_csv(original_file)
    
    # 检查是否有标签列
    if 'label' not in orig_df.columns:
        print(f"警告: {original_file} 没有label列，无法计算正确率")
        return 0.0
    
    # 确保预测结果和原始文件的行数一致
    if len(pred_df) != len(orig_df):
        print(f"警告: 预测结果和原始文件行数不一致，无法计算正确率")
        return 0.0
    
    # 合并数据，只保留有真实标签的行
    merged_df = pd.merge(orig_df, pred_df, on='encoding', suffixes=('_orig', '_pred'))
    
    # 确保合并后的数据不为空
    if merged_df.empty:
        print(f"警告: 合并后的数据为空，无法计算正确率")
        return 0.0
    
    # 提取真实标签和预测标签
    true_labels = merged_df['label_orig'].values
    pred_labels = merged_df['predicted_label_pred'].values
    
    # 计算所有识别出的点数（预测标签不为0的点数）
    identified_points = np.sum(pred_labels != 0)
    
    # 如果没有识别出任何点数，正确率为0
    if identified_points == 0:
        print(f"注意: 没有识别出任何点数")
        return 0.0
    
    # 计算识别正确的点数（预测标签不为0且真实标签也不为0的点数）
    correct_points = np.sum((pred_labels != 0) & (true_labels != 0))
    
    # 计算正确率
    accuracy = correct_points / identified_points
    
    return accuracy

def test_accuracy():
    """测试不同配置的模型准确率
    
    测试不同的窗口大小和滑动窗口步长，计算识别正确率
    正确率定义为：识别正确的点数 / 所有识别出的点数
    只在训练数据集上进行预测和准确率计算
    """
    print("\n=== 开始准确率测试 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 配置参数
    DIMENSIONS = 10000
    TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
    OUTPUT_DIR = Path("accuracy_test_results")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 测试配置：仅使用最佳配置
    test_configs = [
        {'window': 128, 'step': 64, 'n_gram': 4},
    ]
    
    # 用于存储测试结果
    test_results = []
    
    try:
        # 获取所有训练数据文件
        train_files = []
        for ext in ['*.json', '*.xxc.json']:
            train_files.extend(TRAIN_DATA_DIR.glob(ext))
        
        # 过滤掉标注文件
        train_files = [f for f in train_files if '标注' not in f.name]
        
        if not train_files:
            print("错误: 没有找到训练数据文件")
            return False
        
        # 为每个测试配置训练模型并计算准确率
        for config in test_configs:
            window = config['window']
            step = config['step']
            n_gram = config['n_gram']
            
            print(f"\n--- 测试配置: 窗口大小={window}, 滑动步长={step}, N_GRAM_SIZE={n_gram} ---")
            
            # 1. 数据转换：训练数据
            print("\n1. 转换训练数据...")
            converter = UltrasonicDataConverter()
            
            # 转换训练数据
            for data_file in train_files:
                if '标注' in data_file.name:
                    continue
                
                annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
                annotation_path = str(annotation_file) if annotation_file.exists() else None
                
                output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
                
                try:
                    converter.convert_single_file(
                        str(data_file), annotation_path, str(output_csv)
                    )
                except Exception as e:
                    print(f"✗ 转换失败 {data_file.name}: {e}")
                    continue
            
            # 2. 模型训练
            print("\n2. 训练模型...")
            try:
                train_success = train_model(n_gram_size=n_gram)
                if not train_success:
                    print("✗ 模型训练失败")
                    continue
            except Exception as e:
                print(f"✗ 模型训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 3. 模型预测（在训练数据集上进行预测）
            print("\n3. 模型预测...")
            try:
                predict_success = predict_model(data_dir=TRAIN_DATA_DIR)
                if not predict_success:
                    print("✗ 模型预测失败")
                    continue
            except Exception as e:
                print(f"✗ 模型预测失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 4. 计算准确率
            print("\n4. 计算准确率...")
            total_accuracy = 0.0
            file_count = 0
            
            for data_file in train_files:
                if '标注' in data_file.name:
                    continue
                
                # 原始CSV文件路径
                original_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
                # 预测结果CSV文件路径
                predicted_csv = data_file.parent / f"{data_file.stem.replace('_hdc', '')}_hdc.csv"
                
                if not original_csv.exists() or not predicted_csv.exists():
                    print(f"✗ 缺少文件，无法计算 {data_file.name} 的准确率")
                    continue
                
                # 计算单个文件的准确率
                accuracy = calculate_accuracy(predicted_csv, original_csv)
                total_accuracy += accuracy
                file_count += 1
                
                print(f"  {data_file.name}: 准确率 = {accuracy:.4f}")
            
            # 计算平均准确率
            if file_count > 0:
                avg_accuracy = total_accuracy / file_count
            else:
                avg_accuracy = 0.0
            
            print(f"\n  平均准确率: {avg_accuracy:.4f}")
            
            # 保存测试结果
            test_results.append({
                'window': window,
                'step': step,
                'n_gram': n_gram,
                'accuracy': avg_accuracy
            })
        
        # 5. 输出测试结果
        print("\n" + "="*50)
        print("=== 准确率测试结果汇总 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print("=== 准确率测试结果汇总 ===")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        print("配置				准确率")
        print("-"*50)
        
        for result in test_results:
            print(f"窗口={result['window']}, 步长={result['step']}, N_GRAM={result['n_gram']}	{result['accuracy']:.4f}")
        
        # 6. 输出结果
        if test_results:
            best_result = test_results[0]  # 只有一种配置，直接使用
            print("\n" + "="*50)
            print("=== 测试结果 ===")
            print(f"使用的窗口大小: {best_result['window']}")
            print(f"使用的滑动步长: {best_result['step']}")
            print(f"使用的N_GRAM_SIZE: {best_result['n_gram']}")
            print(f"准确率: {best_result['accuracy']:.4f}")
        
        # 7. 保存测试结果到文件
        results_df = pd.DataFrame(test_results)
        results_file = OUTPUT_DIR / "accuracy_test_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n测试结果已保存到: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数，运行所有测试"""
    print("=== 超声波HDC测试套件 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 选择测试类型
    print("请选择测试类型:")
    print("1. 完整管道测试")
    print("2. 准确率测试")
    
    choice = input("输入选择 (1或2): ").strip()
    
    if choice == "1":
        success = test_full_pipeline()
    elif choice == "2":
        success = test_accuracy()
    else:
        print("无效选择，退出测试")
        success = False
    
    if success:
        print("\n✓ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()