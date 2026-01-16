#!/usr/bin/env python3
"""测试完整的超声波HDC管道"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# 添加项目路径
sys.path.append('.')

# 导入超声波HDC程序
from ultrasonic_hdc_emg_style import train_model, predict_model
from ultrasonic_data_converter import UltrasonicDataConverter

# 常量定义
TRAIN_DATA_DIR = Path(r"d:\HDC\data\train")
TEST_DATA_DIR = Path(r"d:\HDC\data\test")

def get_data_files(data_dir):
    """获取数据目录下的所有数据文件，过滤掉标注文件"""
    data_files = []
    for ext in ['*.json', '*.xxc.json']:
        data_files.extend(data_dir.glob(ext))
    return [f for f in data_files if '标注' not in f.name]


def convert_data_files(data_files, converter):
    """转换数据文件为CSV格式"""
    csv_files = []
    for data_file in data_files:
        annotation_file = data_file.parent / f"{data_file.stem}标注{data_file.suffix}"
        annotation_path = str(annotation_file) if annotation_file.exists() else None
        output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
        
        try:
            result_file = converter.convert_single_file(str(data_file), annotation_path, str(output_csv))
            csv_files.append(str(output_csv))
            print(f"✓ 转换完成: {data_file.name} {'(使用标注文件)' if annotation_path else '(无标注文件)'}")
        except Exception as e:
            print(f"✗ 转换失败 {data_file.name}: {e}")
            continue
    return csv_files


def clean_old_files(data_dirs):
    """清理旧的hdc.json和hdc.csv文件"""
    for data_dir in data_dirs:
        print(f"清理目录: {data_dir}")
        # 删除hdc.csv文件
        for file_path in data_dir.glob("*_hdc.csv"):
            try:
                file_path.unlink()
                print(f"✓ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path.name}: {e}")
        # 删除hdc.json文件
        for file_path in data_dir.glob("*_hdc.json"):
            try:
                file_path.unlink()
                print(f"✓ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path.name}: {e}")


def convert_to_hdc_json(data_dirs, eps=5.0, min_samples=5):
    """将CSV文件转换为HDC JSON格式"""
    from csv_to_hdc_json import CsvToHdcJsonConverter
    converter = CsvToHdcJsonConverter()
    
    for data_dir in data_dirs:
        print(f"将 {data_dir} 目录下的CSV文件转换为hdc.json格式")
        converter.batch_convert(str(data_dir), eps=eps, min_samples=min_samples)


def calculate_accuracy(predicted_file, original_file):
    """计算识别正确率
    
    正确率定义为：识别正确的点数 / 所有识别出的点数
    支持单通道和多通道标签
    
    Args:
        predicted_file: 包含预测结果的CSV文件路径
        original_file: 原始CSV文件路径（包含真实标签）
    
    Returns:
        accuracy: 正确率，范围0-1
    """
    try:
        # 读取预测结果文件
        pred_df = pd.read_csv(predicted_file)
        # 读取原始文件
        orig_df = pd.read_csv(original_file)
        
        # 检查是否有多通道标签
        has_multichannel_labels = all([f'label_{i}' in orig_df.columns for i in range(9)])
        has_single_label = 'label' in orig_df.columns
        
        if not has_multichannel_labels and not has_single_label:
            print(f"警告: {original_file} 没有label或label_0到label_8列，无法计算正确率")
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
        
        total_correct = 0
        total_identified = 0
        
        if has_multichannel_labels:
            # 处理多通道标签
            print(f"  使用多通道标签计算准确率")
            
            # 遍历每个通道
            for channel in range(9):
                # 提取真实标签和预测标签
                true_col = f'label_{channel}_orig'
                pred_col = f'predicted_label_{channel}_pred'
                
                # 检查列是否存在
                if true_col not in merged_df.columns or pred_col not in merged_df.columns:
                    continue
                
                true_labels = merged_df[true_col].values
                pred_labels = merged_df[pred_col].values
                
                # 计算当前通道识别出的点数
                channel_identified = np.sum(pred_labels != 0)
                total_identified += channel_identified
                
                # 计算当前通道识别正确的点数
                if channel_identified > 0:
                    channel_correct = np.sum((pred_labels != 0) & (true_labels != 0))
                    total_correct += channel_correct
        else:
            # 处理单通道标签
            print(f"  使用单通道标签计算准确率")
            
            # 提取真实标签和预测标签
            true_labels = merged_df['label_orig'].values
            pred_col = 'predicted_label_pred' if 'predicted_label_pred' in merged_df.columns else 'predicted_label'
            pred_labels = merged_df[pred_col].values
            
            # 计算所有识别出的点数（预测标签不为0的点数）
            total_identified = np.sum(pred_labels != 0)
            
            # 如果没有识别出任何点数，正确率为0
            if total_identified == 0:
                print(f"注意: 没有识别出任何点数")
                return 0.0
            
            # 计算识别正确的点数（预测标签不为0且真实标签也不为0的点数）
            total_correct = np.sum((pred_labels != 0) & (true_labels != 0))
        
        # 计算正确率
        if total_identified == 0:
            print(f"注意: 没有识别出任何点数")
            return 0.0
            
        accuracy = total_correct / total_identified
        
        return accuracy
    except Exception as e:
        print(f"计算准确率时出错: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def test_full_pipeline():
    """测试完整管道"""
    print("=== 开始完整超声波HDC管道测试 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

    # 配置参数
    OUTPUT_DIR = Path("pipeline_test_results")
    TEMP_DIR = Path("pipeline_temp")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    try:
        # 1. 数据转换：训练数据和测试数据
        print("\n=== Step 1: 数据转换 ===")
        converter = UltrasonicDataConverter()
        
        # 转换训练数据
        train_files = get_data_files(TRAIN_DATA_DIR)
        train_csv_files = convert_data_files(train_files, converter)
        
        if not train_csv_files:
            print("[FAIL] 没有成功转换的训练数据文件")
            return False
        
        # 转换测试数据
        test_files = get_data_files(TEST_DATA_DIR)
        test_csv_files = convert_data_files(test_files, converter)
        
        if not test_csv_files:
            print("[FAIL] 没有成功转换的测试数据文件")
            return False
        
        # 2. 模型训练
        print("\n=== Step 2: 模型训练 ===")
        print(f"使用 {TRAIN_DATA_DIR} 目录下的所有CSV文件进行训练")
        
        try:
            if not train_model():
                print("[FAIL] 模型训练失败")
                return False
            print("✓ 模型训练成功")
        except Exception as e:
            print(f"[FAIL] 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. 模型预测
        print("\n=== Step 3: 模型预测 ===")
        print(f"使用 {TEST_DATA_DIR} 目录下的所有CSV文件进行预测")
        
        try:
            if not predict_model():
                print("[FAIL] 模型预测失败")
                return False
            print("✓ 模型预测完成")
        except Exception as e:
            print(f"[FAIL] 模型预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. CSV到HDC JSON转换
        print("\n=== Step 4: CSV到HDC JSON转换 ===")
        try:
            convert_to_hdc_json([TEST_DATA_DIR, TRAIN_DATA_DIR])
            print("✓ CSV到HDC JSON转换完成")
        except Exception as e:
            print(f"[FAIL] CSV到HDC JSON转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. 结果输出
        print("\n=== Step 5: 结果输出 ===")
        print("预测结果已保存到每个测试文件对应的_hdc.csv文件中")
        print("HDC JSON文件已保存到每个测试文件对应的_hdc.json文件中")
        
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
        # 清理临时目录
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)

def test_accuracy():
    """测试不同配置的模型准确率
    
    测试不同的窗口大小和滑动窗口步长，计算识别正确率
    正确率定义为：识别正确的点数 / 所有识别出的点数
    使用train数据训练，对test和train数据都进行预测
    """
    print("\n=== 开始准确率测试 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 配置参数
    OUTPUT_DIR = Path("accuracy_test_results")
    
    # 测试开始时删除train和test目录下所有hdc.json和hdc.csv文件
    print("\n=== 删除旧的hdc.json和hdc.csv文件 ===")
    clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR])
    print("=== 清理完成 ===")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 测试配置：使用用户要求的配置
    # 窗口大小改为12（根据用户要求，平均11行构成一个伤损或结构，所以使用12行）
    # 滑动步长改为1窗
    test_configs = [
        {'window': 12, 'step': 1, 'n_gram': 4},
    ]
    
    # 用于存储测试结果
    test_results = []
    
    try:
        # 获取所有训练数据文件
        train_files = get_data_files(TRAIN_DATA_DIR)
        
        # 获取所有测试数据文件
        test_files = get_data_files(TEST_DATA_DIR)
        
        if not train_files:
            print("错误: 没有找到训练数据文件")
            return False
        
        if not test_files:
            print("警告: 没有找到测试数据文件，将只使用训练数据测试")
        
        # 合并训练和测试文件用于最终测试
        all_test_files = train_files + test_files
        
        # 为每个测试配置训练模型并计算准确率
        for config in test_configs:
            window = config['window']
            step = config['step']
            n_gram = config['n_gram']
            
            print(f"\n--- 测试配置: 窗口大小={window}, 滑动步长={step}, N_GRAM_SIZE={n_gram} ---")
            
            # 1. 数据转换：训练数据和测试数据
            print("\n1. 数据转换...")
            converter = UltrasonicDataConverter()
            
            # 转换训练数据
            print("\n1.1 转换训练数据...")
            convert_data_files(train_files, converter)
            
            # 转换测试数据
            print("\n1.2 转换测试数据...")
            convert_data_files(test_files, converter)
            
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
            
            # 3. 模型预测：先预测test，再预测train
            print("\n3. 模型预测...")
            
            # 预测测试数据
            if test_files:
                print("\n3.1 预测测试数据...")
                try:
                    predict_success = predict_model(data_dir=TEST_DATA_DIR)
                    if not predict_success:
                        print("✗ 模型预测失败")
                        continue
                except Exception as e:
                    print(f"✗ 模型预测失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 预测训练数据
            print("\n3.2 预测训练数据...")
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
            
            # 合并训练和测试文件
            all_files = train_files + test_files
            
            for data_file in all_files:
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
            avg_accuracy = total_accuracy / file_count if file_count > 0 else 0.0
            
            print(f"\n  平均准确率: {avg_accuracy:.4f}")
            
            # 5. CSV到HDC JSON转换
            print("\n5. CSV到HDC JSON转换...")
            try:
                data_dirs_to_convert = [TRAIN_DATA_DIR]
                if test_files:
                    data_dirs_to_convert.append(TEST_DATA_DIR)
                convert_to_hdc_json(data_dirs_to_convert)
                print(f"✓ CSV到HDC JSON转换完成")
            except Exception as e:
                print(f"[FAIL] CSV到HDC JSON转换失败: {e}")
                import traceback
                traceback.print_exc()
                # 继续执行，不中断测试
            
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
    
    # 直接运行方案2：准确率测试
    success = test_accuracy()
    
    if success:
        print("\n✓ 测试成功完成！")
        # 测试结束后自动运行main.py
        print("\n=== 运行main.py ===")
        os.system("python main.py")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()