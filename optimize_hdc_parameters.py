#!/usr/bin/env python3
"""
HDC参数优化脚本
使用网格搜索优化WINDOW、STEP、N_GRAM_SIZE参数
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import subprocess

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入测试脚本
from test_channel8_only import (
    test_channel8_only,
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    MODEL_PATH,
    DIMENSIONS,
    NUM_LEVELS,
    BATCH_SIZE,
    DEFAULT_WINDOW,
    DEFAULT_STEP,
    DEFAULT_N_GRAM_SIZE
)

# 参数网格定义
PARAM_GRID = {
    'window': [10, 15, 20, 25, 30, 40],
    'step': [5, 10, 15, 20, 25],
    'n_gram': [2, 4, 6, 8, 10, 12, 16]
}

# 定义全局变量（用于run_single_test函数）
WINDOW = None
STEP = None
N_GRAM_SIZE = None

def run_single_test(window, step, n_gram):
    """运行单次测试"""
    # 临时修改全局参数
    original_window = DEFAULT_WINDOW
    original_step = DEFAULT_STEP
    original_n_gram = DEFAULT_N_GRAM_SIZE
    
    # 修改全局参数
    global WINDOW, STEP, N_GRAM_SIZE
    WINDOW = window
    STEP = step
    N_GRAM_SIZE = n_gram
    
    try:
        # 清理旧文件
        print(f"\n{'='*60}")
        print(f"测试参数: WINDOW={window}, STEP={step}, N_GRAM_SIZE={n_gram}")
        
        # 运行测试
        success = test_channel8_only()
        
        # 恢复原始参数
        WINDOW = original_window
        STEP = original_step
        N_GRAM_SIZE = original_n_gram
        
        return success
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def grid_search():
    """网格搜索所有参数组合"""
    results = []
    
    print(f"\n{'='*60}")
    print("开始网格搜索参数优化...")
    print(f"{'='*60}")
    
    total_tests = 0
    for window in PARAM_GRID['window']:
        for step in PARAM_GRID['step']:
            for n_gram in PARAM_GRID['n_gram']:
                total_tests += 1
                print(f"\n{'='*60}")
                print(f"测试 {total_tests}/{len(PARAM_GRID['window']) * len(PARAM_GRID['step']) * len(PARAM_GRID['n_gram'])}")
                print(f"参数: WINDOW={window}, STEP={step}, N_GRAM_SIZE={n_gram}")
                
                success = run_single_test(window, step, n_gram)
                
                if success:
                    # 读取结果文件
                    results_file = Path("channel8_test_results/channel8_accuracy_results.csv")
                    if results_file.exists():
                        df = pd.read_csv(results_file)
                        if not df.empty:
                            latest_result = df.iloc[-1]
                            results.append({
                                'window': window,
                                'step': step,
                                'n_gram': n_gram,
                                'precision': latest_result['precision'],
                                'recall': latest_result['recall'],
                                'f1': latest_result['f1'],
                                'f2': latest_result['f2']
                            })
                else:
                    print(f"  测试失败")
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        
        # 找出F1分数最高的参数组合
        best_f1_idx = results_df['f1'].idxmax()
        best_f1 = results_df.iloc[best_f1_idx]
        
        # 找出F2分数最高的参数组合
        best_f2_idx = results_df['f2'].idxmax()
        best_f2 = results_df.iloc[best_f2_idx]
        
        # 找出精确率和召回率都较高的参数组合
        results_df['balanced_score'] = (results_df['precision'] + results_df['recall']) / 2
        best_balanced_idx = results_df['balanced_score'].idxmax()
        best_balanced = results_df.iloc[best_balanced_idx]
        
        # 保存完整结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path(f"channel8_test_results/optimization_results_{timestamp}.csv")
        results_file.parent.mkdir(exist_ok=True)
        results_df.to_csv(results_file, index=False)
        
        print(f"\n{'='*60}")
        print("网格搜索完成！")
        print(f"总测试次数: {total_tests}")
        print(f"成功测试次数: {len(results)}")
        print(f"\n最佳F1分数参数:")
        print(f"  WINDOW={best_f1['window']}, STEP={best_f1['step']}, N_GRAM_SIZE={best_f1['n_gram']}")
        print(f"  精确率={best_f1['precision']:.4f}, 召回率={best_f1['recall']:.4f}, F1分数={best_f1['f1']:.4f}")
        
        print(f"\n最佳F2分数参数（探伤场景黄金指标）:")
        print(f"  WINDOW={best_f2['window']}, STEP={best_f2['step']}, N_GRAM_SIZE={best_f2['n_gram']}")
        print(f"  精确率={best_f2['precision']:.4f}, 召回率={best_f2['recall']:.4f}, F2分数={best_f2['f2']:.4f}")
        
        print(f"\n最佳平衡参数:")
        print(f"  WINDOW={best_balanced['window']}, STEP={best_balanced['step']}, N_GRAM_SIZE={best_balanced['n_gram']}")
        print(f"  精确率={best_balanced['precision']:.4f}, 召回率={best_balanced['recall']:.4f}")
        
        print(f"\n完整结果已保存到: {results_file}")
        print(f"{'='*60}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HDC参数优化脚本')
    parser.add_argument('--mode', type=str, default='grid', 
                       choices=['grid', 'fixed_window', 'optimize_step'],
                       help='优化模式: grid(网格搜索), fixed_window(固定窗口), optimize_step(优化步长)')
    parser.add_argument('--window', type=int, default=20, help='窗口大小（仅用于fixed_window模式）')
    parser.add_argument('--step', type=int, default=20, help='滑动步长（仅用于fixed_window模式）')
    parser.add_argument('--n_gram', type=int, default=4, help='N-gram大小（仅用于fixed_window模式）')
    
    args = parser.parse_args()
    
    if args.mode == 'grid':
        grid_search()
    elif args.mode == 'fixed_window':
        run_single_test(args.window, args.step, args.n_gram)
    elif args.mode == 'optimize_step':
        # 固定WINDOW=20, N_GRAM_SIZE=4，优化STEP
        for step in PARAM_GRID['step']:
            run_single_test(20, step, 4)

if __name__ == '__main__':
    main()
