#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练、测试和打开上位机的脚本
"""

import subprocess
import sys
import os
import time
from datetime import datetime


def run_train():
    """
    运行训练脚本
    """
    print("\n" + "="*50)
    print("开始训练HDC模型")
    print("="*50)
    
    # 运行训练脚本
    result = subprocess.run([sys.executable, "hdc_train.py"], 
                           capture_output=True, 
                           text=True, 
                           cwd=".")
    
    print("\n" + "="*50)
    print("训练结果")
    print("="*50)
    print(result.stdout)
    
    if result.stderr:
        print("\n" + "="*50)
        print("训练错误")
        print("="*50)
        print(result.stderr)
    
    return result.returncode == 0


def find_latest_model():
    """
    查找最新的模型文件
    """
    import glob
    model_dir = "models"
    model_files = glob.glob(os.path.join(model_dir, "hdc_model_*.pkl"))
    if not model_files:
        return None
    # 按修改时间排序，获取最新的模型文件
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


def run_test():
    """
    运行测试脚本
    """
    print("\n" + "="*50)
    print("开始测试HDC模型")
    print("="*50)
    
    # 查找最新的模型文件
    latest_model = find_latest_model()
    if not latest_model:
        print("错误：没有找到模型文件")
        return False
    
    print(f"使用模型：{latest_model}")
    
    # 运行测试脚本
    result = subprocess.run([sys.executable, "hdc_test.py", latest_model], 
                           capture_output=True, 
                           text=True, 
                           cwd=".")
    
    print("\n" + "="*50)
    print("测试结果")
    print("="*50)
    print(result.stdout)
    
    if result.stderr:
        print("\n" + "="*50)
        print("测试错误")
        print("="*50)
        print(result.stderr)
    
    return result.returncode == 0


def open_gui():
    """
    打开上位机
    """
    print("\n" + "="*50)
    print("打开上位机")
    print("="*50)
    
    # 运行GUI程序
    result = subprocess.Popen([sys.executable, "main.py"], 
                            cwd=".")
    
    print("上位机已打开")
    return True


def main():
    """
    主流程
    """
    print("HDC算法一键训练、测试和打开上位机脚本")
    print(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 1. 运行训练
    if not run_train():
        print("\n训练失败，终止流程")
        return 1
    
    # 2. 运行测试
    if not run_test():
        print("\n测试失败，终止流程")
        return 1
    
    # 3. 打开上位机
    if not open_gui():
        print("\n打开上位机失败")
        return 1
    
    print("\n" + "="*50)
    print("流程完成！")
    print("="*50)
    print("下一步优化方向：")
    print("1. 优化HDC算法参数，提高模型准确率")
    print("2. 增强训练数据，添加更多标注样本")
    print("3. 改进测试流程，添加更详细的结果分析")
    print("4. 优化上位机界面，提高用户体验")
    print("5. 添加模型性能对比功能")
    print("6. 实现模型版本管理")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
