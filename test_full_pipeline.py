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
            
            # 检查是否已存在CSV文件
            output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            if output_csv.exists():
                print(f"✓ CSV文件已存在，跳过转换: {data_file.name}")
                train_csv_files.append(str(output_csv))
                continue
            
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
            
            # 检查是否已存在CSV文件
            output_csv = data_file.parent / f"{data_file.stem}_hdc.csv"
            if output_csv.exists():
                print(f"✓ CSV文件已存在，跳过转换: {data_file.name}")
                test_csv_files.append(str(output_csv))
                continue
            
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

if __name__ == "__main__":
    success = test_full_pipeline()
    if success:
        print("\n✓ 测试成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 测试失败！")
        sys.exit(1)